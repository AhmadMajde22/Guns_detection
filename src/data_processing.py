import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custome_exception import CustomException

# Initialize logger
logger = get_logger(__name__)

class GunDataset(Dataset):
    def __init__(self, root: str):
        logger.info(f"Initializing GunDataset with root directory: {root}")
        try:
            self.images_path = os.path.join(root, 'Images')
            self.labels_path = os.path.join(root, 'Labels')

            # Validate directories exist
            if not os.path.exists(self.images_path):
                logger.error(f"Images directory not found: {self.images_path}")
                raise FileNotFoundError(f"Images directory not found: {self.images_path}")
            if not os.path.exists(self.labels_path):
                logger.error(f"Labels directory not found: {self.labels_path}")
                raise FileNotFoundError(f"Labels directory not found: {self.labels_path}")

            self.img_name = sorted(os.listdir(self.images_path))
            self.label_name = sorted(os.listdir(self.labels_path))

            logger.info(f"Successfully initialized dataset with {len(self.img_name)} images")
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {str(e)}", exc_info=True)
            raise CustomException("Dataset initialization failed", e)

    def __getitem__(self, idx):
        try:
            # logger.info(f"Processing item at index: {idx}")

            image_path = os.path.join(self.images_path, self.img_name[idx])
            # logger.debug(f"Loading image from: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image at {image_path}")
                raise IOError(f"Unable to load image at {image_path}")

            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            img_res = img_rgb / 255.0
            img_res = torch.as_tensor(img_res).permute(2, 0, 1)

            label_filename = os.path.splitext(self.img_name[idx])[0] + ".txt"
            label_path = os.path.join(self.labels_path, label_filename)
            # logger.debug(f"Loading label from: {label_path}")

            if not os.path.exists(label_path):
                logger.error(f"Label file not found: {label_path}")
                raise FileNotFoundError(f"Label file not found: {label_path}")

            target = {
                "boxes": torch.tensor([]),
                "area": torch.tensor([]),
                "image_id": torch.tensor([idx]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

            with open(label_path, 'r') as f:
                l_count = int(f.readline().strip())
                box = [list(map(int, f.readline().split())) for _ in range(l_count)]

            if box:
                area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
                labels = [1] * len(box)  # class label = 1 for all

                target['boxes'] = torch.tensor(box, dtype=torch.float32)
                target['area'] = torch.tensor(area, dtype=torch.float32)
                target['labels'] = torch.tensor(labels, dtype=torch.int64)

            if torch.cuda.is_available():
                img_res = img_res.cuda()
                for key in target:
                    target[key] = target[key].cuda()

            # logger.info(f"Successfully processed item {idx}")
            return img_res, target

        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}", exc_info=True)
            raise CustomException(f"Failed to load data for index {idx}", e)

    def __len__(self):
        return len(self.img_name)


if __name__ == "__main__":
    root_path = r"artifacts/raw"

    dataset = GunDataset(root=root_path)
    image, target = dataset[0]

    print("Image Shape : ", image.shape)
    print("Target Keys : ", target.keys())
    print("Bounding boxes :", target['boxes'])
