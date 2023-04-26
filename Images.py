import tensorflow as tf
import pandas as pd
import json
import os
import shutil
import matplotlib.pyplot as plt

#k+1 softmax layers where the last layer predicts the len of numbers in the house number

# Set Image Directory (change this to accomodate yourself)
main_dir = '/Users/blakecurtsinger/Desktop/BZAN554/Group_Assignments/group_assignment_3/SVHN'
train_dir = main_dir + '/train_cropped'
test_dir = main_dir + '/test_cropped'

## Metadata

# rename .json files
os.rename(main_dir + "/train/digitStruct.json", main_dir + "/train/digitStruct_train.json")
os.rename(main_dir + "/test/digitStruct.json", main_dir + "/test/digitStruct_test.json")

# Set directories to move .json files
source_train = main_dir + "/train/digitStruct_train.json"
source_test = main_dir + "/test/digitStruct_test.json"
destination = main_dir + "/metadata"

# Make metadata folder, move .json files
os.makedirs(destination, exist_ok=True)
shutil.move(source_train, destination)
shutil.move(source_test, destination)

# Load JSON training file with image metadata
with open(os.path.join(main_dir, 'metadata', 'digitStruct_train.json'), 'r') as f:
    metadata_train = json.load(f)

# Load JSON test file with image metadata
with open(os.path.join(main_dir, 'metadata', 'digitStruct_test.json'), 'r') as f:
    metadata_test = json.load(f)

#Function for getting image labels. Returns a dictionary with the keys as the filenames and their associated labels
def get_image_labels(metadata):
    image_labels = {}
    for obj in metadata:
        filename = obj['filename']
        num_digits = len(obj['boxes'])
        label = [int(obj['boxes'][i]['label']) for i in range(num_digits)]
        image_labels[filename] = label
    return image_labels

train_labels = get_image_labels(metadata_train)
test_labels = get_image_labels(metadata_test)

train_labels.values

# Height and Width for generator
img_height = 256
img_width = 256

# Image generators
train_generator = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode='grayscale',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode='grayscale',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Image shape
for image_batch in test_generator:
  print(image_batch.shape)
  break

#This shows that the generator works. Plot 9 training images
plt.figure(figsize=(10, 10))
for images in train_generator.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")
plt.show()

## Pre-processing and Modeling

inputs = tf.keras.layers.Input(shape=(img_height, img_width, 1), name='input') 
#rescale input layer between 0 and 1
rescaled = tf.keras.layers.Rescaling(scale=1./255)(inputs)
#Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(rescaled)
#MaxPooling2D: pool_size is window size over which to take the max
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=256,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
#dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(64, activation = 'relu')(x)
yhat1 = tf.keras.layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputs, outputs = yhat)
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
history = model.fit(
    train_generator,
    epochs = 100,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = 5,
            restore_best_weights = True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            patience = 3
        )
    ]
)
