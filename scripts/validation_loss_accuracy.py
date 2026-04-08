import matplotlib.pyplot as plt

history = {
    "loss": [...],
    "val_loss": [...],
    "accuracy": [...],
    "val_accuracy": [...]
}

plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Loss")
plt.savefig("../results/model/learning_curves.png")