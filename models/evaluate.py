
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# PATHS

test_dir = r"C:\Users\inamm\Desktop\HandGesturepptproject\dataset_split\test"

# Create results folder
os.makedirs("results", exist_ok=True)


# DATA GENERATOR

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False   # IMPORTANT
)


# LOAD MODEL

model = tf.keras.models.load_model("models/best_model.h5")
# EVALUATE MODEL
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# PREDICTIONS

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# CLASSIFICATION REPORT

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)


# CONFUSION MATRIX (PRINT)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)



# CONFUSION MATRIX (PLOT + SAVE)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
print("Saved: results/confusion_matrix.png")
plt.close()


# ACCURACY BAR GRAPH

plt.figure()
plt.bar(["Test Accuracy"], [accuracy])
plt.title("Model Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)   # important for visibility
plt.tight_layout()
plt.savefig("results/test_accuracy.png")
print("Saved: results/test_accuracy.png")
plt.close()


# PER CLASS ACCURACY GRAPH

class_correct = cm.diagonal()
class_total = cm.sum(axis=1)
class_accuracy = class_correct / class_total

plt.figure()
plt.bar(class_names, class_accuracy)
plt.xticks(rotation=45)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/per_class_accuracy.png")
print("Saved: results/per_class_accuracy.png")
plt.close()