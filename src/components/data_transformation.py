import os
import sys
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifacts


class DataTransformation:
    """
    Handles audio → spectrogram transformation:
    - Loads .wav files from train/test folders
    - Converts them into spectrogram images
    - Saves spectrograms into artifacts directories
    """

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact) -> None:
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
        except Exception as e:
            raise CustomException(e, sys)

    def load_audio_files(self, folder_path: str, label: str):
        """
        Loads all .wav files from a folder and returns dataset list.
        Uses soundfile backend to avoid TorchCodec dependency issues.
        """
        try:
            dataset = []
            walker = sorted(str(p) for p in Path(folder_path).glob('*.wav'))
            if not walker:
                logging.warning(f"No .wav files found in {folder_path}")
                return dataset

            for file_path in walker:
                # Use soundfile to read audio
                data, sample_rate = sf.read(file_path, dtype="float32")
                waveform = torch.tensor(data).unsqueeze(0)  # add channel dimension
                dataset.append([waveform, sample_rate, label, file_path])
            return dataset
        except Exception as e:
            raise CustomException(e, sys)

    def create_spectrogram_images(self, dataloader, label_dir, is_test=False):
        """
        Converts audio waveforms into spectrogram images and saves them.
        """
        try:
            base_dir = self.data_transformation_config.test_dir if is_test else self.data_transformation_config.images_dir
            directory = os.path.join(base_dir, label_dir)
            os.makedirs(directory, exist_ok=True)

            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=400, window_fn=torch.hann_window
            ).to(self.device)

            spectrogram_count = 0
            for i, data in enumerate(dataloader):
                waveform = data[0].to(self.device)
                spectrogram_tensor = spectrogram_transform(waveform)

                # Convert to log scale and handle NaN/inf
                spectrogram_data = spectrogram_tensor[0].log2()
                spectrogram_data = torch.where(
                    torch.isfinite(spectrogram_data),
                    spectrogram_data,
                    torch.tensor(0.0, device=self.device)
                )

                # Ensure 2D array for imsave
                arr = spectrogram_data.detach().cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[0]  # take first channel
                arr = arr.squeeze()

                path_to_save_img = os.path.join(directory, f"spec_img{i}.png")
                plt.imsave(path_to_save_img, arr, cmap='viridis')
                spectrogram_count += 1

            logging.info(f"Generated {spectrogram_count} spectrograms for {label_dir} in {directory}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Main entry point:
        - Loads train/test audio files for dog and cat
        - Generates spectrograms
        - Returns artifact paths
        """
        try:
            logging.info("Initiating data transformation...")

            # Paths to train/test folders
            train_dog_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'train', 'dog')
            train_cat_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'train', 'cat')
            test_dog_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'test', 'dog')
            test_cat_path = os.path.join(self.data_ingestion_artifact.data_folder_path, 'test', 'cat')

            # Load datasets
            train_dog = self.load_audio_files(train_dog_path, 'dog')
            train_cat = self.load_audio_files(train_cat_path, 'cat')
            test_dog = self.load_audio_files(test_dog_path, 'dog')
            test_cat = self.load_audio_files(test_cat_path, 'cat')

            if not train_dog or not train_cat or not test_dog or not test_cat:
                raise CustomException("One of the datasets is empty. Check folder structure.", sys)

            # Create DataLoaders
            trainloader_dog = torch.utils.data.DataLoader(train_dog, batch_size=1, shuffle=SHUFFLE)
            trainloader_cat = torch.utils.data.DataLoader(train_cat, batch_size=1, shuffle=SHUFFLE)
            testloader_dog = torch.utils.data.DataLoader(test_dog, batch_size=1, shuffle=False)
            testloader_cat = torch.utils.data.DataLoader(test_cat, batch_size=1, shuffle=False)

            # Generate spectrograms
            self.create_spectrogram_images(trainloader_dog, 'dog', is_test=False)
            self.create_spectrogram_images(trainloader_cat, 'cat', is_test=False)
            self.create_spectrogram_images(testloader_dog, 'dog', is_test=True)
            self.create_spectrogram_images(testloader_cat, 'cat', is_test=True)

            # Return artifact
            data_transformation_artifact = DataTransformationArtifacts(
                images_folder_path=self.data_transformation_config.images_dir,
                test_folder_path=self.data_transformation_config.test_dir
            )
            logging.info("Data transformation completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
