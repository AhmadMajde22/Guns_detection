import os
import torch
import time
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from src.model_architecture import FasetRCNNModel
from src.custome_exception import CustomException
from src.data_processing import GunDataset
from src.logger import get_logger
from config.model_training_config import *
from torch.utils.tensorboard import SummaryWriter


logger = get_logger(__name__)
model_save_path = MODEL_SAVE_PATH
os.makedirs(model_save_path, exist_ok=True)


class ModelTraining:
    def __init__(self, model_class, num_classes, learning_rate, epochs, dataset_path, device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"tensorboard_logs/{timestamp}"

        os.makedirs(self.log_dir,exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        try:
            logger.info(f"Initializing model with {self.num_classes} classes on {self.device}")
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(self.device)
            logger.info("Model successfully moved to device")

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.info(f"Optimizer initialized with learning rate: {self.learning_rate}")

        except Exception as e:
            logger.error("Failed to initialize model and optimizer", exc_info=True)
            raise CustomException("Error while initializing model training.", e)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def split_dataset(self):
        try:
            dataset = GunDataset(self.dataset_path)
            logger.info(f"Total dataset size: {len(dataset)}")
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            logger.info(f"Splitting dataset into {train_size} training and {val_size} validation samples")

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=self.collate_fn)

            logger.info("Dataset split successfully")
            return train_loader, val_loader

        except Exception as e:
            logger.error("Error during dataset splitting", exc_info=True)
            raise CustomException("Error while splitting data.", e)

    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()

            for epoch in range(self.epochs):
                logger.info(f"\n===== Starting Epoch {epoch + 1}/{self.epochs} =====")
                epoch_start_time = time.time()

                # Training phase
                self.model.train()
                total_train_loss = 0.0

                for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)):
                    # Move images and targets to device
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    self.optimizer.zero_grad()
                    losses = self.model(images, targets)

                    if isinstance(losses, dict):
                        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
                    else:
                        total_loss = losses[0]
                        self.writer.add_scalar("Loss/Train",total_loss.item(),epoch * len(train_loader) + i)


                    if total_loss == 0:
                        logger.warning(f"Loss at step {i} is zero. Check data and targets.")
                        raise ValueError("Total loss is zero.")

                    self.writer.add_scalar("Loss/Train",total_loss.item(),epoch * len(train_loader) + i)

                    total_loss.backward()
                    self.optimizer.step()

                    total_train_loss += total_loss.item()
                    # logger.debug(f"Step [{i+1}/{len(train_loader)}] - Loss: {total_loss.item():.4f}")

                avg_train_loss = total_train_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} - Avg Training Loss: {avg_train_loss:.4f}")

                self.writer.flush()

                self.model.train()  # Keep in training mode for validation losses
                total_val_loss = 0.0
                with torch.no_grad():
                    for j, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)):
                        # Move images and targets to device
                        images = [img.to(self.device) for img in images]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                        val_losses = self.model(images, targets)

                        # Handle validation losses (same as training)
                        if isinstance(val_losses, dict):
                            val_loss = sum(v for v in val_losses.values() if isinstance(v, torch.Tensor))
                        elif isinstance(val_losses, (list, tuple)):
                            val_loss = val_losses[0]
                        else:
                            val_loss = val_losses

                        total_val_loss += val_loss.item()
                        # logger.debug(f"Validation Step [{j+1}/{len(val_loader)}] - Loss: {val_loss.item():.4f}")

                    avg_val_loss = total_val_loss / len(val_loader)
                    logger.info(f"Epoch {epoch+1} - Avg Validation Loss: {avg_val_loss:.4f}")

                # Save model checkpoint
                model_path = os.path.join(model_save_path, f"fasterrcnn_epoch{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model checkpoint saved to: {model_path}")

                epoch_duration = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds\n")

        except Exception as e:
            logger.error("Failed during training", exc_info=True)
            raise CustomException("Error while training the model.", e)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    training = ModelTraining(
        model_class=FasetRCNNModel,
        num_classes=2,
        learning_rate=0.0001,
        epochs=10,
        dataset_path=r'artifacts/raw',
        device=device
    )

    training.train()
