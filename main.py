import tensorflow as tf
import numpy as np
import cv2
from train import load_wav_to_nparray

model = tf.keras.models.load_model("model.h5")


def load(file_path):
    log_spectrogram = load_wav_to_nparray(file_path)
    data = np.array([log_spectrogram])

    return data


file_paths = [
    "fan/id_00/normal/00000000.wav",
    "fan/id_00/normal/00000001.wav",
    "fan/id_06/abnormal/00000001.wav",
    "fan/id_06/abnormal/00000002.wav",
]
for file_path in file_paths:
    input_data = load(file_path)
    predictions = model.predict(input_data)
    print(predictions[0][0])
