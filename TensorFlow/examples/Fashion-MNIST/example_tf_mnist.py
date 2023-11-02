#Example of a simple Neural Network with a classification model for computer vision
#we are using the Fashion-MMIST dataset from zalando
print('Neural Network example')
print(' ')
import tensorflow as tf
from tensorflow import keras
#from tensorflow.common.library_loader import load_habana_module
#load_habana_module()

print('Load modules succesful')

fmnist = keras.datasets.fashion_mnist
#load the training and test set
(train_images,train_labels), (test_images,test_labels) = fmnist.load_data()
print('Data loaded')

#normalize the data
train_images=train_images/255.0
test_images = test_images/255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('Model created')

#Train the model
model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images,test_labels)
