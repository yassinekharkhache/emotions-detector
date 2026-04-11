import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../results/model/final_emotion_model.keras")

df = pd.read_csv("../data/test.csv")
X, y = [], []
print(df.columns)
for _, row in df.iterrows():
    pixels = np.array(row[" pixels"].split(), dtype="float32").reshape(48, 48)
    X.append(pixels)
    if "emotion" in df.columns:
        y.append(row["emotion"])


X = np.array(X) / 255.0
X = np.expand_dims(X, -1)

preds = model.predict(X)
pred_labels = np.argmax(preds, axis=1)

if y:
    acc = (pred_labels == np.array(y)).mean()
    print(f"Accuracy on test set: {acc*100:.0f}%")