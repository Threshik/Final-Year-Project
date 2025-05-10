from google.colab import drive
drive.mount('/content/drive')

train_dir = '/content/drive/MyDrive/Indian-monuments/images/train'
test_dir = '/content/drive/MyDrive/Indian-monuments/images/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

import time
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_model(base_model, num_classes):
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model_name, base_model, train_data, test_data, epochs=5):
    print(f"\nTraining {model_name}...")
    model = build_model(base_model, train_data.num_classes)
    start = time.time()
    history = model.fit(train_data, epochs=epochs, validation_data=test_data, verbose=1)
    end = time.time()
    print(f"{model_name} completed in {(end-start)/60:.2f} mins")
    return model, history

from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2

models_dict = {
    "VGG16": VGG16(include_top=False, input_shape=(224,224,3), weights='imagenet'),
    "ResNet50": ResNet50(include_top=False, input_shape=(224,224,3), weights='imagenet'),
    "InceptionV3": InceptionV3(include_top=False, input_shape=(224,224,3), weights='imagenet'),
    "MobileNetV2": MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet'),
}

histories = {}
accuracies = {}

for name, base in models_dict.items():
    model, history = train_and_evaluate(name, base, train_data, test_data, epochs=5)
    histories[name] = history
    accuracies[name] = max(history.history['val_accuracy']) * 100
    model.save(f'/content/drive/MyDrive/Indian-monuments/{name}_model.h5')

def plot_history(histories):
    for name, history in histories.items():
        plt.figure(figsize=(12, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train acc')
        plt.plot(history.history['val_accuracy'], label='val acc')
        plt.title(f'{name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title(f'{name} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

plot_history(histories)

plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.title("Validation Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.grid(True)
plt.show()


# Assuming you have a trained model named 'model'
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 1: Install Gradio (run only once in Colab)
!pip install gradio --quiet

# Step 2: Import Libraries
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Step 3: Load Any One Saved Model (e.g., ResNet50)
model_path = '//content/drive/MyDrive/Indian-monuments/InceptionV3_model.h5'  # Change to VGG16_model.h5 etc. if needed
model = load_model(model_path)

# Step 4: Get Class Names
class_names = list(train_data.class_indices.keys())

# Step 5: Define Prediction Function
def predict_monument(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}


# Step 6: Create Gradio Interface
interface = gr.Interface(
    fn=predict_monument,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Indian Monument Classifier",
    description="Upload an image of a monument to get its classification."
)

# Step 7: Launch the Interface
interface.launch(debug=True, share=True)
