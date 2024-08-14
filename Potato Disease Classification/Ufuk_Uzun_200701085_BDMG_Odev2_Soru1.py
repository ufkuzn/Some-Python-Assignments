import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

class PotatoDiseaseClassifier:
    def __init__(self, data_path, image_size=(224, 224), batch_size=32, epochs=20):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self):
        def custom_preprocessing(image):
            image = tf.image.resize(image, self.image_size)
            image = tf.keras.applications.inception_v3.preprocess_input(image)
            return image

        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            preprocessing_function=custom_preprocessing
        )

        train_generator = train_datagen.flow_from_directory(self.data_path,
                                                            target_size=self.image_size,
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical',
                                                            subset='training')

        validation_generator = train_datagen.flow_from_directory(self.data_path,
                                                                 target_size=self.image_size,
                                                                 batch_size=self.batch_size,
                                                                 class_mode='categorical',
                                                                 subset='validation')

        def lr_schedule(epoch):
            lr = 1e-3
            if epoch > 10:
                lr *= 0.5
            return lr

        callbacks = [LearningRateScheduler(lr_schedule)]

        self.model.fit(train_generator,
                       steps_per_epoch=train_generator.samples // self.batch_size,
                       epochs=self.epochs,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.samples // self.batch_size,
                       callbacks=callbacks)

    def save_model(self, model_path='potato_disease_model.keras'):
        self.model.save(model_path)
        print(f"Model kaydedildi: {model_path}")


data_path = "gorseller"
classifier = PotatoDiseaseClassifier(data_path)
classifier.build_model()
classifier.train_model()
classifier.save_model()
