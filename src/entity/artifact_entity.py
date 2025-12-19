from dataclasses import dataclass

# ---------------- Data Ingestion ----------------
@dataclass
class DataIngestionArtifacts:
    data_folder_path: str   # Path to extracted dataset folder

# --------- Data Transformation ----------------
@dataclass
class DataTransformationArtifacts:
    images_folder_path: str # Path to spectrogram images
    test_folder_path: str   # Path to test data

# ----------- Model Trainer ----------------
@dataclass
class ModelTrainerArtifacts:
    model_path: str                 # Path to trained model
    result: dict                    # Training results (loss, accuracy)
    transformer_object_path: str    # Path to saved preprocessing object

# ---------------- Model Evaluation ------
@dataclass
class ModelEvaluationArtifacts:
    best_model_loss: float          # Loss of best model
    is_model_accepted: bool         # Whether model is accepted
    trained_model_path: str         # Path to trained model
    best_model_path: str            # Path to best model

# ---------------- Model Pusher --
@dataclass
class ModelPusherArtifacts:
    response: dict                  # Response after pushing model locally
