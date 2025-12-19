import os
from dataclasses import dataclass
from from_root import from_root
from src.constants import *

# ---------------- Data Ingestion ----------------
@dataclass
class DataIngestionConfig:
    # Artifact directory for ingestion logs/outputs
    data_ingestion_artifact_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)

    # Direct dataset paths (no zip/unzip)
    dataset_dir: str = os.path.join(from_root(), DATASET_DIR_NAME)          # cats_dogs root folder
    train_dir: str = os.path.join(dataset_dir, TRAIN_DIR_NAME)              # cats_dogs/train
    test_dir: str = os.path.join(dataset_dir, TEST_DIR_NAME)                # cats_dogs/test


# ---------------- Data Transformation -------
@dataclass
class DataTransformationConfig:
    data_transformation_artifact_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, DATA_TRANSFORMATION_ARTIFACTS_DIR)
    images_dir: str = os.path.join(data_transformation_artifact_dir, IMAGES_DIR)   # spectrograms folder
    test_dir: str = os.path.join(data_transformation_artifact_dir, TEST_DIR)       # transformed test folder


# ---------- Model Trainer ----------------
@dataclass
class ModelTrainerConfig:
    model_trainer_artifact_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
    model_path: str = os.path.join(model_trainer_artifact_dir, MODEL_NAME)                 # trained model path
    transformer_object_path: str = os.path.join(model_trainer_artifact_dir, TRANSFORM_OBJECT_NAME)  # preprocessing object


# ---------------- Model Evaluation ------
@dataclass
class ModelEvaluationConfig:
    model_evaluation_artifacts_dir: str = os.path.join(from_root(), ARTIFACTS_DIR, MODEL_EVALUATION_DIR)
    best_model_dir: str = os.path.join(model_evaluation_artifacts_dir, BEST_MODEL_DIR)
    best_model: str = os.path.join(best_model_dir, BEST_MODEL_NAME)


# ---- Prediction Pipeline ----------------
@dataclass
class PredictionPipelineConfig:
    prediction_artifact_dir: str = os.path.join(from_root(), STATIC_DIR, MODEL_SUB_DIR)
    model_download_path: str = os.path.join(prediction_artifact_dir, MODEL_NAME)            # model for inference
    transforms_path: str = os.path.join(prediction_artifact_dir, TRANSFORM_OBJECT_NAME)     # preprocessing object
    image_path: str = os.path.join(from_root(), STATIC_DIR, UPLOAD_SUB_DIR, IMAGE_NAME)     # uploaded image path
    audio_path_dir: str = os.path.join(from_root(), STATIC_DIR, UPLOAD_SUB_DIR)             # uploaded audio path
