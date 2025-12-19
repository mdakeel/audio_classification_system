import os
import sys
import joblib
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts
from src.entity.custom_model import ResNet9
from src.utils import get_default_device, to_device, DeviceDataLoader, evaluate

class ModelEvaluation:
    """
    Evaluates trained model against local test spectrograms.
    - Loads transform from training
    - Loads test images from artifacts/data_transformation/test
    - Optionally compares against a locally saved 'best model'
    """

    def __init__(self,
                 model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifacts,
                 model_trainer_artifacts: ModelTrainerArtifacts) -> None:
        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifacts = model_trainer_artifacts

    def get_test_data_loader(self, test_data: ImageFolder):
        """
        Builds dataloader for test dataset.
        """
        try:
            logging.info("Preparing test DataLoader")
            test_dl = DataLoader(test_data, BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            return test_dl
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model_path(self):
        """
        Returns path to locally stored best model if exists.
        """
        try:
            best_model_dir = self.model_evaluation_config.best_model_dir
            best_model_path = self.model_evaluation_config.best_model
            os.makedirs(best_model_dir, exist_ok=True)

            if os.path.exists(best_model_path):
                logging.info(f"Best model found at: {best_model_path}")
                return best_model_path

            logging.info("Best model not found locally; will evaluate current trained model only.")
            return None
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self, dataset: ImageFolder):
        """
        Builds model with correct class count.
        """
        try:
            logging.info("Initializing ResNet9 for evaluation")
            num_classes = len(dataset.classes)
            model = ResNet9(IN_CHANNELS, num_classes)
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self):
        """
        Loads test data, chooses model (best or current), evaluates, and returns loss.
        """
        try:
            # Load transform saved during training
            transformer_object = joblib.load(self.model_trainer_artifacts.transformer_object_path)

            # Load test spectrogram images
            test_root = self.data_transformation_artifact.test_folder_path
            logging.info(f"Loading test images from: {test_root}")
            test_data = ImageFolder(test_root, transform=transformer_object)

            test_dl = self.get_test_data_loader(test_data)

            # Decide which model to evaluate: best (if available) else trained
            best_model_path = self.get_best_model_path()
            model_path_to_use = best_model_path or self.model_trainer_artifacts.model_path

            model = self.get_model(test_data)
            device = get_default_device()
            model = to_device(model, device)

            logging.info(f"Loading model weights from: {model_path_to_use}")
            model.load_state_dict(torch.load(model_path_to_use, map_location=device))
            model.eval()

            logging.info("Moving test data to device")
            test_dl = DeviceDataLoader(test_dl, device)

            logging.info("Evaluating model on test data")
            result = evaluate(model=model, val_loader=test_dl)
            test_loss = result["val_loss"]
            return test_loss
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        """
        Compares test loss to baseline and decides acceptance.
        """
        try:
            test_loss = self.evaluate_model()
            logging.info(f"Test loss: {test_loss}")

            trained_model_loss = self.model_trainer_artifacts.result["val_loss"]
            evaluation_response = (test_loss < trained_model_loss) and (trained_model_loss < BASE_LOSS)

            model_evaluation_artifacts = ModelEvaluationArtifacts(
                s3_model_loss=test_loss,                 # kept field name for compatibility
                is_model_accepted=evaluation_response,
                trained_model_path=os.path.dirname(self.model_trainer_artifacts.model_path),
                s3_model_path=self.model_evaluation_config.best_model_dir  # local best model dir
            )

            logging.info(f"Model evaluation completed: {model_evaluation_artifacts}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
