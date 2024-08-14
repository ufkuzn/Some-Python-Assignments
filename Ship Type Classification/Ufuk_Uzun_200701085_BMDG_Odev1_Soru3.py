import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.optimizers import Adam
from keras.applications import VGG16

# Veri seti dizinleri
train_path = 'C:/Users/Ufuk/PycharmProjects/pythonProjects/ships_dataset/train'
valid_path = 'C:/Users/Ufuk/PycharmProjects/pythonProjects/ships_dataset/valid'
test_path = 'C:/Users/Ufuk/PycharmProjects/pythonProjects/ships_dataset/test'
google_path = 'C:/Users/Ufuk/PycharmProjects/pythonProjects/new'

class_names = ['Aircraft Carrier', 'Bulkers', 'Car Carrier', 'Container Ship', 'Cruise',
               'DDG', 'Recreational', 'Sailboat', 'Submarine', 'Tug']

# Gemi veri kümenizi yükleyin ve ön işleme tabi tutun
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Eğitim verileri
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(64, 64),
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    classes=class_names)

# Doğrulama verileri
valid_generator = test_datagen.flow_from_directory(valid_path,
                                                   target_size=(64, 64),
                                                   batch_size=64,
                                                   class_mode='categorical',
                                                   classes=class_names)

# Test verileri
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(64, 64),
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  classes=class_names)

# Transfer öğrenme modeli oluştur
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Önceki katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()
model.add(base_model)

model.add(layers.Flatten())
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=10, activation='softmax'))

# Modeli derleme
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme (ince ayar)
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=valid_generator,
                    batch_size=64)

# Kendi topladığınız verilerle modeli eğitme
google_datagen = ImageDataGenerator(rescale=1. / 255)
google_generator = google_datagen.flow_from_directory(google_path,
                                                      target_size=(64, 64),
                                                      batch_size=64,
                                                      class_mode='categorical',
                                                      classes=class_names)

model.fit(google_generator, epochs=5)

# Test verileri üzerinde modeli değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print("Fine-tuned Test Accuracy:", test_accuracy)

# Modeli kaydetme
model.save('ship_classifier_model.h5')

# Eğitim sonuçlarını görselleştirme
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
