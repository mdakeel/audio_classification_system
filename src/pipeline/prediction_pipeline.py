import os, sys
import joblib
import torch
import soundfile as sf
import torchaudio
from PIL import Image
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.constants import *
from src.entity.config_entity import PredictionPipelineConfig
from src.utils import get_default_device, to_device, predict_image
from src.entity.custom_model import ResNet9

DEVICE = get_default_device()


class SinglePrediction:
    """
    Handles single audio file prediction:
    - Loads trained model locally
    - Converts audio to spectrogram image
    - Applies transform and predicts class
    """

    def __init__(self):
        try:
            self.prediction_config = PredictionPipelineConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        """
        Loads the trained ResNet9 model from local storage.
        """
        try:
            prediction_model_path = self.prediction_config.model_download_path
            if not os.path.exists(prediction_model_path):
                raise FileNotFoundError(f"Model file not found at {prediction_model_path}")

            model = to_device(ResNet9(IN_CHANNELS, NUM_CLASSES), DEVICE)
            model.load_state_dict(torch.load(prediction_model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def get_audio_waveform_sr(self, filename):
        """
        Loads audio waveform from a .wav file using soundfile.
        """
        try:
            data, sample_rate = sf.read(filename, dtype="float32")
            waveform = torch.tensor(data).unsqueeze(0)  # add channel dimension
            return waveform, sample_rate
        except Exception as e:
            raise CustomException(e, sys)

    def create_spectrogram_images(self):
        """
        Converts audio file in upload folder to spectrogram image.
        """
        try:
            audio_path_dir = self.prediction_config.audio_path_dir
            filename = None
            for file in os.listdir(audio_path_dir):
                if file.endswith(".wav"):
                    filename = os.path.join(audio_path_dir, file)
                    break

            if filename is None:
                raise FileNotFoundError("No .wav file found in upload directory")

            waveform, _ = self.get_audio_waveform_sr(filename=filename)

            image_save_path = self.prediction_config.image_path
            spectrogram = torchaudio.transforms.Spectrogram()(waveform)

            # Convert to log scale and ensure 2D array
            spec_data = spectrogram.log2()[0]
            spec_data = torch.where(
                torch.isfinite(spec_data),
                spec_data,
                torch.tensor(0.0)
            )
            arr = spec_data.detach().cpu().numpy()
            arr = arr.squeeze()

            plt.imsave(image_save_path, arr, cmap='viridis')
        except Exception as e:
            raise CustomException(e, sys)

    def _get_image_tensor(self, image_path):
        """
        Loads spectrogram image and applies saved transform.
        """
        try:
            img = Image.open(image_path)
            transforms = joblib.load(self.prediction_config.transforms_path)
            img_tensor = transforms(img)
            return img_tensor
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self):
        """
        Full prediction pipeline:
        - Load model
        - Create spectrogram from audio
        - Transform image
        - Predict class
        """
        try:
            model = self.get_model()
            self.create_spectrogram_images()
            image = self._get_image_tensor(self.prediction_config.image_path)
            result = predict_image(image, model, DEVICE)
            return result
        except Exception as e:
            raise CustomException(e, sys)
