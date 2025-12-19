import os
import sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    """
    Handles local data ingestion:
    - Verifies dataset folder exists locally
    - Returns artifact paths for downstream components
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
            # Path where artifacts will be stored
            self.data_ingestion_artifact = self.data_ingestion_config.data_ingestion_artifact_dir
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Main entry point:
        - Verifies dataset folder exists
        - Returns artifact with dataset path
        """
        try:
            logging.info("Initiating local data ingestion...")
            os.makedirs(self.data_ingestion_artifact, exist_ok=True)

            # Dynamic dataset path from config (cats_dogs folder)
            dataset_path = self.data_ingestion_config.dataset_dir

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")

            logging.info(f"Dataset found at {dataset_path}")

            # Return artifact pointing to dataset folder
            data_ingestion_artifact = DataIngestionArtifacts(
                data_folder_path=dataset_path
            )

            logging.info("Data ingestion completed successfully.")
            return data_ingestion_artifact

        except Exception as e:
            logging.error(f"Failed to complete data ingestion: {str(e)}")
            raise CustomException(e, sys)
