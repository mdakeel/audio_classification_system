import os
import torch
from datetime import datetime

# Root directory where all artifacts will be stored
ARTIFACTS_DIR: str = "artifacts"

# ---------------- Data Ingestion ----------------
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion"
DATASET_DIR_NAME: str = "cats_dogs"        # Main dataset folder
TRAIN_DIR_NAME: str = "train"              # Training data folder
TEST_DIR_NAME: str = "test"                # Testing data folder

# ---------------- Data Transformation ----------------
DATA_TRANSFORMATION_ARTIFACTS_DIR: str = "data_transformation"
IMAGES_DIR: str = "spectrograms"           # Folder for spectrogram images
TEST_DIR: str = "test"                     # Folder for transformed test split
SHUFFLE = True
PIN_MEMORY = True
NUM_WORKERS = 0

# ---------------- Model Training ----------------
MODEL_TRAINER_ARTIFACTS_DIR: str = "model_training"
MODEL_NAME: str = "model.pt"                  # Saved model file
TRANSFORM_OBJECT_NAME: str = "transform.pkl"  # Preprocessing transform object
BATCH_SIZE: int = 15
EPOCHS: int = 15
LEARNING_RATE: float = 0.001
GRAD_CLIP: float = 0.1
WEIGHT_DECAY: float = 1e-4
IN_CHANNELS: int = 3
OPTIMIZER = torch.optim.RMSprop
NUM_CLASSES: int = 2                          # Cat vs Dog

# ----- Model Evaluation ----------------
MODEL_EVALUATION_DIR: str = "model_evaluation"
BEST_MODEL_DIR: str = "best_model"
BEST_MODEL_NAME: str = "model.pt"
BASE_LOSS: float = 1.00

# ---------------- Prediction Pipeline --
PREDICTION_PIPELINE_DIR_NAME: str = "prediction_artifact"
IMAGE_NAME: str = "image.jpg"
STATIC_DIR: str = "static"
MODEL_SUB_DIR: str = "model"
UPLOAD_SUB_DIR: str = "upload"
