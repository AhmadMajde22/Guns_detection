# Guns Detection Project

## Overview

This project implements a gun detection system using deep learning, specifically utilizing the Faster R-CNN architecture. The system is designed to detect and localize guns in images, making it potentially useful for security and surveillance applications.

## Project Structure

```
├── artifacts/
│   ├── models/                  # Trained model checkpoints
│   │   └── fasterrcnn_*.pth    # Model checkpoints for each epoch (1-25)
│   └── raw/                    # Raw dataset directory
│       ├── Images/             # Input images for training/testing
│       └── Labels/             # Corresponding annotation files
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── data_ingestion_config.py  # Data pipeline settings
│   └── model_training_config.py  # Model and training hyperparameters
├── logs/                       # Application logging directory
│   └── log_*.log              # Daily rotating log files
├── notebook/                   # Jupyter notebooks for analysis
│   └── notebook.ipynb         # Main development notebook
├── src/                       # Source code directory
│   ├── __init__.py
│   ├── custome_exception.py   # Custom error handling
│   ├── data_ingestion.py     # Data loading and preparation
│   ├── data_processing.py    # Data preprocessing utilities
│   ├── logger.py            # Logging configuration
│   ├── model_architecture.py # FasterRCNN model definition
│   └── model_training.py    # Training pipeline implementation
├── tensorboard_logs/         # Training visualization data
│   └── */                   # Date-wise tensorboard event files
├── utils/                    # Utility functions
│   └── helpers.py           # Helper functions and utilities
├── DVC_Bucket/              # DVC remote storage
├── Guns_Detection/          # Virtual environment
├── dvc.yaml                 # DVC pipeline configuration
├── dvc.lock                 # DVC pipeline state
├── main.py                  # Main execution script
├── requirements.txt         # Project dependencies
└── setup.py                # Package installation setup
```

### Key Components Description

1. **Data Management (`artifacts/`)**:
   - Raw data storage with separate directories for images and labels
   - Model checkpoints saved for each training epoch
   - DVC tracked for version control

2. **Configuration (`config/`)**:
   - Centralized configuration management
   - Separate configs for data pipeline and model training
   - Easy modification of hyperparameters

3. **Source Code (`src/`)**:
   - Modular implementation of each component
   - Custom exception handling
   - Structured logging system
   - Complete training pipeline implementation

4. **Monitoring and Logging**:
   - Daily rotating logs in `logs/`
   - TensorBoard integration for:
     - Loss visualization
     - Performance metrics
     - Model architecture

5. **Development Tools**:
   - Jupyter notebooks for experimentation
   - DVC for data and model versioning
   - Virtual environment for dependency isolation

6. **Utility Functions (`utils/`)**:
   - Reusable helper functions
   - Common processing utilities

## Features

- Object detection using Faster R-CNN architecture
- Data version control using DVC
- MLOps pipeline implementation
- TensorBoard integration for training visualization
- Comprehensive logging system
- Modular code structure with configuration management

## Technologies Used

- Python
- PyTorch
- DVC (Data Version Control)
- TensorBoard
- FastAPI
- MLOps tools

## Setup and Installation

1. Create and activate virtual environment:

```bash
python -m venv Guns_Detection
source Guns_Detection/Scripts/activate  # On Windows use: .\Guns_Detection\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Setup DVC:

```bash
dvc init
dvc remote add -d storage DVC_Bucket
```

## Usage

### Data Preparation

The data should be organized in the `artifacts/raw` directory with separate folders for Images and Labels.

### Training

To train the model:

```bash
python main.py
```

### Monitoring

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir tensorboard_logs
```

## Detailed Components Description

### Source Code (`src/`)

#### Data Ingestion (`data_ingestion.py`)

- Handles dataset loading and preprocessing
- Functions:
  - `get_data()`: Loads images and annotations
  - `preprocess_data()`: Converts annotations to model format
  - `create_dataloader()`: Creates PyTorch DataLoader objects
- Integration with DVC for data versioning
- Custom dataset class implementation for PyTorch

#### Model Architecture (`model_architecture.py`)

- Implements Faster R-CNN with ResNet-50 backbone
- Components:
  - Region Proposal Network (RPN)
  - ROI Pooling layer
  - Detection heads for classification and box regression
- Custom anchor box generation
- Feature pyramid network implementation
- Transfer learning setup with pretrained weights

#### Training Pipeline (`model_training.py`)

- Complete training loop implementation
- Features:
  - Learning rate scheduling
  - Gradient clipping
  - Early stopping mechanism
  - Model checkpointing
  - Multi-GPU support
  - Mixed precision training
- Loss functions:
  - Classification loss (Cross Entropy)
  - Bounding box regression loss (Smooth L1)
  - RPN losses

#### Exception Handling (`custome_exception.py`)

- Custom exception classes for:
  - Data loading errors
  - Model configuration errors
  - Training pipeline errors
- Detailed error tracking and logging

#### Logger (`logger.py`)

- Comprehensive logging system
- Features:
  - Daily rotating file handlers
  - Custom log formatting
  - Different log levels for various components
  - TensorBoard integration

#### Data Processing (`data_processing.py`)

- Image preprocessing utilities
- Augmentation pipeline:
  - Random horizontal flip
  - Color jittering
  - Normalization
- Annotation format conversion
- Batch processing utilities

### Configuration System (`config/`)

#### Data Ingestion Config (`data_ingestion_config.py`)

```python
class DataIngestionConfig:
    dataset_path: str
    train_ratio: float = 0.8
    batch_size: int = 16
    num_workers: int = 4
    image_size: tuple = (800, 800)
    augmentation_params: dict
```

#### Model Training Config (`model_training_config.py`)

```python
class ModelTrainingConfig:
    epochs: int = 25
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.1
    checkpoint_path: str
```

### Utility Functions (`utils/helpers.py`)

- Visualization tools for:
  - Bounding box drawing
  - Training progress plots
  - Confusion matrix generation
- Metric calculations:
  - Mean Average Precision (mAP)
  - Intersection over Union (IoU)
  - Precision-Recall curves
- Data conversion utilities
- Model evaluation helpers

### DVC Pipeline (`dvc.yaml`)

```yaml
stages:
  data_load:
    cmd: python src/data_ingestion.py
    deps:
      - src/data_ingestion.py
      - artifacts/raw
    outs:
      - artifacts/processed

  train:
    cmd: python src/model_training.py
    deps:
      - src/model_training.py
      - artifacts/processed
    params:
      - model_training_config.py:epochs
      - model_training_config.py:learning_rate
    outs:
      - artifacts/models
    metrics:
      - metrics.json
```

### Tensorboard Integration (`tensorboard_logs/`)

- Real-time monitoring of:
  - Training/validation losses
  - Learning rate changes
  - Model gradients
  - Performance metrics
  - GPU utilization
  - Memory usage

### Notebook Development (`notebook/notebook.ipynb`)

- Experimental analysis
- Model prototyping
- Data exploration
- Performance visualization
- Error analysis
- Hyperparameter tuning studies

## Logs

Training logs are stored in the `logs/` directory with daily rotation.

## Model Performance and Metrics

### Training Details

- **Training Duration**: 25 epochs
- **Optimization**: Adam optimizer with learning rate scheduling
- **Batch Size**: 16 images per batch
- **Hardware**: GPU-accelerated training

### Model Checkpointing

- Checkpoints saved at each epoch (`artifacts/models/`)
- Best model selection based on:
  - Mean Average Precision (mAP)
  - Intersection over Union (IoU)
  - Loss convergence

### Performance Metrics

- **Object Detection**:
  - Average Precision (AP) @ IoU=0.5
  - Average Precision (AP) @ IoU=0.75
  - Mean Average Precision (mAP)
- **Training Metrics**:
  - Classification accuracy
  - Regression loss
  - RPN performance
