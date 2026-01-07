import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

#We will train Convolutional Neural Network on MNIST dataset.

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape = (28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')

])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    x_train,y_train, epochs = 5
)
loss, accuracy = model.evaluate(x_test,y_test)
print("Loss = ",loss)
print("Accuracy = ",accuracy)

plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['loss'], label = 'Loss')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

index = 10
test_image = x_test[index]
label_test = y_test[index]

reshape_test_input = test_image.reshape(1,28,28,1)
prediction = model.predict(reshape_test_input)
predicted_label = np.argmax(prediction)
plt.figure(figsize=(10,10))
plt.imshow(test_image, cmap = 'gray')
plt.title('Predicted Label: ' + str(predicted_label))
plt.show()