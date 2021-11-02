from tensorflow import keras
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Invariants (Don't change on your own)
images_loc = "pants_images"
labels_name = "labels.csv"
label_column = "Pants/Shorts"
image_dim = (400, 400)  # (width, height)
stretch_images = True  # Stretch or crop images to fit aspect ratio
k_folds = 5

# Parameters (Change to improve model)
epochs = 5
pixel_range = (0, 1)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = "adam"

# Model (Change to improve model)
# Different layers can be researched at https://www.tensorflow.org/api_docs/python/tf/keras/layers
# Default model contains four layers, excluding the inputs:
# Flatten - Convert image array to a 1-D vector
#   input_shape determines the inputs of the network
# Dense with 128 nodes - Standard neural network layer
# Dropout - While training, turns off inputs of previous layer
#   20% of the time to prevent overfitting
# Dense with 2 nodes - Output layer with no activation function applied
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=image_dim))  # Do not change
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2))  # Final layer outputs = number of labels


def resize(image, dimensions):
    old_width, old_height = image.size

    # Original is too tall
    if old_width / old_height < dimensions[0] / dimensions[1]:
        # Crop image
        new_height = old_width * dimensions[1] / dimensions[0]
        image = image.crop((0, (old_height - new_height) / 2,
                            old_width, (old_height + new_height) / 2))

        # Resize image
        return image.resize(dimensions)

    else:  # Original is too wide
        # Crop image
        new_width = old_height * dimensions[0] / dimensions[1]
        image = image.crop(((old_width - new_width) / 2, 0,
                            (old_width + new_width) / 2, old_height))

        # Resize image
        return image.resize(dimensions)


# Import labels
labels = pd.read_csv(labels_name)

# Import images
images = []
gray_images = []

for name in labels["Data Number"]:
    images.append(Image.open(images_loc + "/" + str(name) + ".jpg"))

    # Resize image
    if stretch_images:
        images[-1] = images[-1].resize(image_dim)

    else:
        images[-1] = resize(images[-1], image_dim)

    # Convert image to grayscale
    gray_images.append(ImageOps.grayscale(images[-1]))

    # Convert images to numpy array
    images[-1] = np.array(images[-1])
    gray_images[-1] = np.array(gray_images[-1])

# Scale pixel brightnesses
for i in range(len(images)):
    images[i] = images[i] * (pixel_range[1] - pixel_range[0]) / 255.0
    images[i] += pixel_range[0]
    gray_images[i] = gray_images[i] * (pixel_range[1] - pixel_range[0]) / 255.0
    gray_images[i] += pixel_range[0]

# Specify particular label category
labels = labels[label_column]

# Convert labels to numbers for training
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

# Create training and test sets
# Data is divided into k_folds groups
# For 0 <= i < k_folds, group i is the test set, while everything else is the training set
# Each element of fold_indices is of the form (train_indices, test_indices)
# Training image set number i is then np.array([gray_images[index] for index in fold_indices[i][0]])
# Training label set number i is then np.array([labels[index] for index in fold_indices[i][0]])
# Test image set number i is then np.array([gray_images[index] for index in fold_indices[i][1]])
# Test label set number i is then np.array([labels[index] for index in fold_indices[i][1]])
fold_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True)
fold_indices = []

for i in fold_splitter.split(gray_images, labels):
    fold_indices.append(i)

for i in range(len(fold_indices)):
    np.random.shuffle(fold_indices[i][0])
    np.random.shuffle(fold_indices[i][1])

# Define the training regimine the model will go through
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Train the model
responses = []
answer_key = []

for i in range(k_folds):
    X_train = np.array([gray_images[index] for index in fold_indices[i][0]])
    y_train = np.array([labels[index] for index in fold_indices[i][0]])
    X_test = np.array([gray_images[index] for index in fold_indices[i][1]])
    y_test = np.array([labels[index] for index in fold_indices[i][1]])

    model.fit(X_train, y_train, epochs=epochs)

    responses = np.append(responses, model.predict(X_test).argmax(axis=1))
    answer_key = np.append(answer_key, y_test)

# Evaluate the model
print("\nAccuracy: {}".format(accuracy_score(answer_key, responses)))

# Calculate confusion matrix
class_names = encoder.classes_
conf_mat = pd.DataFrame(np.transpose(confusion_matrix(
    answer_key, responses, labels=encoder.transform(class_names))), index=class_names, columns=class_names)
print("Confusion Matrix (Predicted x Actual):")
print(conf_mat, "\n")

"""# Make a new model which outputs probabilities of the input being in each class
probability_model = keras.Sequential([model, keras.layers.Softmax()])

# Set output to normal notation (not scientific notation)
np.set_printoptions(suppress=True)

# Create a dictionary with each predicted probability for the 0th piece of data
output = probability_model(np.array(gray_images[:1])).numpy()[0]
output = {encoder.inverse_transform([i])[0]: output[i]
          for i in range(len(output))}
print(output)"""
