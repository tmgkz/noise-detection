import wave

import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import io
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

LOAD_FILE_COUNT = 50


def load_wav_to_nparray(file_path="fan/id_00/normal/00000000.wav") -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    stft = librosa.stft(y, n_fft=512, hop_length=256)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


if __name__ == "__main__":
    positives = []
    negatives = []
    for i in range(LOAD_FILE_COUNT):
        print(f"loading...{i+1}/{LOAD_FILE_COUNT}")
        positive_arr = load_wav_to_nparray(f"fan/id_00/normal/{str(i).zfill(8)}.wav")
        positives.append(positive_arr)
        # negatives.append(np.random.rand(*positive_arr.shape))
        negatives.append(
            load_wav_to_nparray(f"fan/id_00/abnormal/{str(i).zfill(8)}.wav")
        )

    positive_label = np.zeros(np.array(positives).shape[0])
    negative_label = np.ones(np.array(negatives).shape[0])
    x = np.concatenate((positives, negatives))
    y = np.concatenate((positive_label, negative_label))

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    # モデルを構築
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=X_train.shape[1:]
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                1, activation="sigmoid"
            ),  # 0 1判断だけでいいのでSigmoid
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # モデルを学習
    epochs = 10  # エポック数
    batch_size = 32  # バッチサイズ
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
    )

    # モデルを評価
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    model.save("model.h5")
