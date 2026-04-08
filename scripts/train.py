import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Load data
df = pd.read_csv("../data/train.csv")

X, y = [], []
for _, row in df.iterrows():
    pixels = np.array(row["pixels"].split(), dtype="float32").reshape(48, 48)
    X.append(pixels)
    y.append(row["emotion"])

X = np.array(X) / 255.0
X = np.expand_dims(X, -1)
y = tf.keras.utils.to_categorical(y, 7)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.TensorBoard(log_dir="../results/logs")
]

# Train
model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2, callbacks=cb)

# Save
model.save("../results/model/final_emotion_model.keras")
model.summary()