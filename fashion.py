# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#loading data bro....
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#preprocess the data commented coz it's not involved in code but gives an idea how things work
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
#plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



#building model(neural network) with layers ## Designing the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Flatten: makes 28*28 pixels 2d array into 784pixels 1d array, it actually reformats 2d array data to 1d array 
# Dense: this is a fully connceted layer.
# The first Dense layer has 128 nodes (or neurons). 
# The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
# Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.

#compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#train the model
model.fit(train_images, train_labels, epochs=5)

#Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# Making Predictions

predictions = model.predict(test_images)

# Sample Prediction of an test_images index 0
predictions[0]

# A prediction is an array of 10 numbers.
#  These describe the "confidence" of the model that the image corresponds to each of the 10 different articles of clothing.
#  We can see which label has the highest confidence value:

np.argmax(predictions[0])
# Displays the same confidence
test_labels[0]

# Graphing this confidence using function plot_value_array
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label]),color=color)


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# 0th index
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# 12th index
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()




# Grab an image from the test dataset
img = test_images[0]

#print(img.shape)  gives (28,28)


# tf.keras models are optimized to make predictions on a batch, or collection, of examples at once.
#  So even though we're using a single image, we need to add it to a list:
# Add the image to a batch where it's the only member.

# indirectly we are adding a image to a list as tf.keras model accepts a list instead of single image
img = (np.expand_dims(img,0))

# print(img.shape)  gives(1,28,28)

# Predicting the image
predictions_single = model.predict(img)

# print(predictions_single)

# plotting graphs
# plot_value_array(i, predictions_array, true_label)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()


