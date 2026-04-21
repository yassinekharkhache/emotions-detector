import matplotlib.pyplot as plt
import pickle



def plot_learning_curves(history):
    # Loss
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Loss")
    plt.legend(["Train", "Validation"])

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.savefig("../results/model/learning_curves.png")