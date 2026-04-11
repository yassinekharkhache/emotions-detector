from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Load data
df = pd.read_csv("../data/train.csv")

X, y = [], []
for _, row in df.iterrows():
    pixels = np.array(row["pixels"].split(), dtype="uint8").reshape(48, 48)
    X.append(pixels)
    y.append(row["emotion"])

X = np.array(X) / 255.0
X = np.expand_dims(X, -1)
y = tf.keras.utils.to_categorical(y, 7)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
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
history = model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2, callbacks=cb)

# Save
model.save("../results/model/final_emotion_model.keras")

with open("../results/model/history.pkl", "wb") as f:
    pickle.dump(history.history, f)
model.summary()