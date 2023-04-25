import tensorflow as tf
import pandas as pd
import json
import os
import shutil
import matplotlib.pyplot as plt

# Set Image Directory (change this to accomodate yourself)
main_dir = '/Users/blakecurtsinger/Desktop/BZAN554/Group_Assignments/group_assignment_3/SVHN'
train_dir = main_dir + '/train'
test_dir = main_dir + '/test'

## Metadata

# rename .json files
os.rename(train_dir + "/digitStruct.json", train_dir + "/digitStruct_train.json")
os.rename(test_dir + "/digitStruct.json", test_dir + "/digitStruct_test.json")

# Set directories to move .json files
source_train = train_dir + "/digitStruct_train.json"
source_test = test_dir + "/digitStruct_test.json"
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

train_labels["1.png"]


# Image generators
train_generator = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels=train_labels,
    label_mode="int",
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=test_labels,
    label_mode="int",
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

#This shows that the generator works. Plot 9 training images
plt.figure(figsize=(10, 10))
for images, labels in train_generator.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")   
plt.show()

# the directory containing the images


# iterate over the dictionary
for image_filename, labels in train_labels.items():
    # load the image
    image_path = os.path.join(train_dir, image_filename)
    image = plt.imread(image_path)

    # display the image and its labels
    plt.imshow(image)
    plt.title(f"Labels: {labels}")
    plt.show()

