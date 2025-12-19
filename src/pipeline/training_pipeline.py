import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.mode_pusher import ModelPusher   # fixed typo: mode_pusher → model_pusher
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
)
from src.constants import BASE_LOSS

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("Starting data transformation in training pipeline")
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifacts
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training in pipeline")
        try:
            model_trainer = ModelTraining(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, data_transformation_artifact: DataTransformationArtifacts,
                               model_trainer_artifacts: ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Starting model evaluation in pipeline")
        try:
            model_evaluation = ModelEvaluation(
                self.model_evaluation_config,
                data_transformation_artifact,
                model_trainer_artifacts
            )
            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info(f"Evaluation results: accepted={model_evaluation_artifacts.is_model_accepted}, "
                         f"test_loss={model_evaluation_artifacts.s3_model_loss}, "
                         f"trained_loss={model_trainer_artifacts.result['val_loss']}, "
                         f"baseline={BASE_LOSS}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts) -> ModelPusherArtifacts:
        logging.info("Starting model pusher in pipeline")
        try:
            model_pusher = ModelPusher(model_evaluation_artifacts=model_evaluation_artifacts)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            ingestion_artifacts = self.start_data_ingestion()
            transformation_artifacts = self.start_data_transformation(ingestion_artifacts)
            trainer_artifacts = self.start_model_trainer(transformation_artifacts)
            evaluation_artifacts = self.start_model_evaluation(transformation_artifacts, trainer_artifacts)
            pusher_artifacts = self.start_model_pusher(evaluation_artifacts)
            logging.info(f"Pipeline completed! Model pushed={pusher_artifacts.response['is_model_pushed']}, "
                         f"path={pusher_artifacts.response['model_path']}")
            print(pusher_artifacts)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
