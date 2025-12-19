import os
import sys
import shutil
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelPusherArtifacts

class ModelPusher:
    """
    Handles pushing trained model to local 'production' folder.
    - If model is accepted, copy trained model into best_model directory
    - Otherwise, skip and log message
    """

    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        self.model_evaluation_artifacts = model_evaluation_artifacts
    
    def initiate_model_pusher(self):
        try:
            logging.info("Initiating model pusher component")

            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                best_model_dir = self.model_evaluation_artifacts.s3_model_path  # reused field name for compatibility

                # Ensure best_model_dir exists
                os.makedirs(best_model_dir, exist_ok=True)

                # Destination path for promoted model
                dest_model_path = os.path.join(best_model_dir, MODEL_NAME)

                # Verify trained model exists
                if not os.path.exists(trained_model_path):
                    raise FileNotFoundError(f"Trained model path {trained_model_path} does not exist")

                # Copy model into production folder
                shutil.copy2(os.path.join(trained_model_path, MODEL_NAME), dest_model_path)

                message = "Model Pusher promoted the current trained model to local production storage"
                response = {
                    "is_model_pushed": True,
                    "model_path": dest_model_path,
                    "message": message
                }
                logging.info(f"Model push response: {response}")
            else:
                best_model_dir = self.model_evaluation_artifacts.s3_model_path
                os.makedirs(best_model_dir, exist_ok=True)

                message = "Current trained model not accepted; production model has better loss"
                response = {
                    "is_model_pushed": False,
                    "model_path": best_model_dir,
                    "message": message
                }
                logging.info(f"Model push response: {response}")

            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            logging.info(f"Model pusher completed! Artifacts: {model_pusher_artifacts}")
            return model_pusher_artifacts

        except Exception as e:
            logging.error(f"Error in model pusher: {str(e)}")
            raise CustomException(e, sys)
