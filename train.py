import wave

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import io
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


def load_wav_to_nparray(file_path="fan/id_00/normal/00000000.wav") -> np.ndarray:
    with wave.open(file_path, "rb") as wf:
        channels = wf.getnchannels()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        buffer = wf.readframes(n_frames)

    data = np.frombuffer(buffer, dtype="int16")

    # ステレオ音源の場合はモノラルに変換
    if channels == 2:
        data = data[::2] + data[1::2]

    # スペクトログラムを計算
    frequencies, times, spectrogram_data = spectrogram(data, framerate)

    fig, ax = plt.subplots()
    ax.pcolormesh(times, frequencies, np.log10(spectrogram_data), shading="gouraud")
    ax.set_axis_off()
    fig.tight_layout()
    ax.set_box_aspect(1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    arr_resize = np.resize(open_cv_image, (470, 470))

    return arr_resize


if __name__ == "__main__":
    x_positives = []
    x_negatives = []
    for i in range(5):
        print(f"loading...{i+1}/50")
        arr = load_wav_to_nparray(f"fan/id_00/normal/{str(i).zfill(8)}.wav")
        x_positives.append(arr)
        x_negatives.append(np.random.rand(*arr.shape))
    x_positives = np.stack([x_positives, x_positives, x_positives], axis=-1)
    x_negatives = np.stack([x_negatives, x_negatives, x_negatives], axis=-1)

    height, width = arr.shape[:2]
    y_positive = np.zeros(x_positives.shape[0])
    y_negative = np.ones(x_negatives.shape[0])
    x = np.concatenate([x_positives, x_negatives], axis=0)
    y = np.concatenate([y_positive, y_negative], axis=0)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2

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
        validation_data=(X_val, y_val),
    )

    # モデルを評価
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    model.save("anomaly_detection_model.h5")
