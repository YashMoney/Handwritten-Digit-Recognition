import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Display the shape of the test set
print(X_test.shape)

# Visualize some sample images from the dataset
plt.imshow(X_test[0])
plt.show()

plt.imshow(X_train[2])
plt.show()

# Normalize the data by scaling pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Display the first image data in the training set after normalization
print(X_train[0])

# Define the model
model = Sequential()

# Add an Input layer to define the input shape
model.add(Input(shape=(28, 28)))

# Flatten the input data (from 28x28 to 784)
model.add(Flatten())

# Add a Dense layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Add a Dense layer with 32 neurons and ReLU activation
model.add(Dense(32, activation='relu'))

# Add the output layer with 10 neurons (for the 10 classes) and softmax activation
model.add(Dense(10, activation='softmax'))

# Print the model summary
model.summary()

# Compile the model with loss, optimizer, and metrics
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Train the model for 25 epochs with 20% validation split
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)

# Predict the probabilities on the test set
y_prob = model.predict(X_test)

# Convert the probabilities to class predictions
y_pred = y_prob.argmax(axis=1)

# Display the predictions
print(y_pred)

# Calculate and display the accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Visualize another test image
plt.imshow(X_test[1])
plt.show()

# Predict the class of the second test image
predicted_class = model.predict(X_test[1].reshape(1, 28, 28)).argmax(axis=1)
print(predicted_class)
