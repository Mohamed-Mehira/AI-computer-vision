import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
    keras.layers.Softmax()
])

print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False) # in case there's no softmax layer in the model, from_logits should be True
# in case of one_hot_encoded labels (y), we use normal categorecal-cross-entropy
metrics = ["accuracy"]

model.compile(optimizer, loss, metrics)

model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)

model.evaluate(x_test, y_test, batch_size=64, verbose=2)

predictions = model(x_test)
# predictions = model.predict(x_test, batch_size=64)
# predictions = tf.nn.softmax(predictions)  # if there's no softmax layer in the original model
predictions = np.argmax(predictions, axis=1)
labels = y_test

plt.imshow(x_test[7], cmap='gray')
plt.show()
print('prediction: ', predictions[7])
print('label: ', labels[7])


# import sys; sys.exit()

## 1) Save whole model
# model.save("nn.h5")  # HDF5 format
# model.save("neural_net")  # SavedModel format
# new_model = keras.models.load_model("nn.h5")


## 2) Save only weights
# model.save_weights("nn_weights.h5")
# model.load_weights("nn_weights.h5")


## 3) Save only architecture (to_json)
# json_string = model.to_json()

# with open("nn_model", "w") as f:
#     f.write(json_string)

# with open("nn_model", "r") as f:
#     loaded_json_string = f.read()

# new_model = keras.models.model_from_json(loaded_json_string)
