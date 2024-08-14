import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

def load_dataset(directory):
    labels = []
    images = []
    label_encoder = LabelEncoder()

    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                img = Image.open(img_path).resize((150, 150))
                img = np.array(img)
                if img.shape == (150, 150, 3):
                    images.append(img)
                    labels.append(subdir)

    labels = label_encoder.fit_transform(labels)
    images = np.array(images)
    return images, to_categorical(labels), label_encoder.classes_

def create_pretrained_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))  # Load MobileNetV2
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
    x = Dense(256, activation='relu')(x)  # Add a fully-connected layer
    x = Dropout(0.5)(x)  # Add a dropout layer
    predictions = Dense(num_classes, activation='softmax')(x)  # Add a logistic layer
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers of MobileNetV2
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X, y, class_names = load_dataset('dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
train_generator = datagen.flow(X_train, y_train, batch_size=32)

model = create_pretrained_model(len(class_names))
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('final_model.keras')

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Performance')

plt.show()
