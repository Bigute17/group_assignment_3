import pandas as pd
import json
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import transform
import seaborn as sns
import h5py

# Set Image Directory (change this to accomodate yourself)
main_dir = '/Users/blakecurtsinger/Desktop/BZAN554/Group_Assignments/group_assignment_3/SVHN'
train_dir = main_dir + '/train'
test_dir = main_dir + '/test'

## PREP METADATA

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
    

## VISUALIZE BOUNDING BOXES
def display_boxes(img, bounding_boxes):
    """Displays an image and overlays the bounding boxes
    """
    # Opens and identifies the given image file
    image = Image.open(img)
    
    # Use draw module can be used to annotate the image
    draw = ImageDraw.Draw(image)
    
    for b in bounding_boxes:
        
        # Bounding box rectangle [x0, y0, x1, y1]
        rectangle = [b['left'], b['top'], b['left'] + b['width'], b['top'] + b['height']]
        
        # Draw a rectangle on top of the image
        draw.rectangle(rectangle, outline="blue")
        
    # Return altered image 
    image.show()
       
    return image


# Select an image and the corresponding boxes
image = train_dir + '/1.png'
image_bounding_boxes = metadata_train[0]['boxes']
     
# Display image with bounding boxes
display_boxes(image, image_bounding_boxes)


## CONVERT TO PANDAS DF
def dict_to_df(image_bounding_boxes, path):
    """ Helper function for flattening the bounding box dictionary
    """
    # Store each bounding box
    boxes = []
    
    # For each set of bounding boxes
    for image in image_bounding_boxes:
        
        # For every bounding box
        for bbox in image['boxes']:
            
            # Store a dict with the file and bounding box info
            boxes.append({
                    'filename': path + "/" + image['filename'],
                    'label': bbox['label'],
                    'width': bbox['width'],
                    'height': bbox['height'],
                    'top': bbox['top'],
                    'left': bbox['left']})
            
    # return the data as a DataFrame
    return pd.DataFrame(boxes)


# Save bounding box data to csv
bbox_file = main_dir + '/bounding_boxes.csv'

if not os.path.isfile(bbox_file):
    
    # Extract every individual bounding box as DataFrame  
    train_df = dict_to_df(metadata_train, train_dir)
    test_df = dict_to_df(metadata_test, test_dir)

    print("Training", train_df.shape)
    print("Test", test_df.shape)
    print('')

    # Concatenate all the information in a single file
    df = pd.concat([train_df, test_df])
    
    print("Combined", df.shape)

    # Write dataframe to csv
    df.to_csv(bbox_file, index=False)

    # Delete the old dataframes to save memory
    del train_df, test_df, metadata_train, metadata_test
    
else:
    # Load preprocessed bounding boxes
    df = pd.read_csv(bbox_file)

# Display the first 10 rows of dataframe
df.head()


## GROUP IMAGES BY FILENAME
# Rename the columns to more suitable names
df.rename(columns={'left': 'x0', 'top': 'y0'}, inplace=True)

# Calculate x1 and y1
df['x1'] = df['x0'] + df['width']
df['y1'] = df['y0'] + df['height']

# Perform the following aggregations on the columns
aggregate = {'x0': 'min',
             'y0': 'min',
             'x1': 'max',
             'y1': 'max',
             'label': [('labels', lambda x: list(x)), ('num_digits', 'count')]}


df = df.groupby('filename').agg(aggregate).reset_index()

# Fix the column names after aggregation
df.columns = [x[0] if i < 5 else x[1] for i, x in enumerate(df.columns.values)]

# Display the results
df.head()


##Expand boxes in images
# Before expansion
def display_bbox(image_path, bbox):
    """ Helper function to display a single image and bounding box
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    draw.rectangle([bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']], outline="blue")
    # Return altered image 
    image.show()
    return image


# Select a image and bounding box
image = train_dir + "/1.png"
bbox = df[df["filename"] == image]

# Display image
display_bbox(image, bbox)

# After expansion
# Calculate the increase in both directions
df['x_inc'] = ((df['x1'] - df['x0']) * 0.30) / 2.
df['y_inc'] = ((df['y1'] - df['y0']) * 0.30) / 2.

# Apply the increase in all four directions
df['x0'] = (df['x0'] - df['x_inc']).astype('int')
df['y0'] = (df['y0'] - df['y_inc']).astype('int')
df['x1'] = (df['x1'] + df['x_inc']).astype('int')
df['y1'] = (df['y1'] + df['y_inc']).astype('int')


# Select a image and bounding box
image = train_dir + "/1.png"
bbox = df[df["filename"] == image]

# Display image
display_bbox(image, bbox)

##REPORT IMAGE SIZE
def get_img_size(filepath):
    """Returns the image size in pixels given as a 2-tuple (width, height)
    """
    image = Image.open(filepath)
    return image.size 

def get_img_sizes(folder):
    """Returns a DataFrame with the file name and size of all images contained in a folder
    """
    image_sizes = []
    
    # Get all .png images contained in the folder
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    
    # Get image size of every individual image
    for image in images:
        w, h = get_img_size(folder + "/" + image)
        image_size = {'filename': folder + "/" + image, 'image_width': w, 'image_height': h}
        image_sizes.append(image_size)
        
    # Return results as a pandas DataFrame
    return pd.DataFrame(image_sizes)


# Extract the image sizes
train_sizes = get_img_sizes(train_dir)
test_sizes = get_img_sizes(test_dir)

# Concatenate all the information in a single file
image_sizes = pd.concat([train_sizes, test_sizes])

# Delete old dataframes
del train_sizes, test_sizes

# Display 10 image sizes
image_sizes.head(10)

# Merge the dataframes

print("Bounding boxes", df.shape)
print("Image sizes", image_sizes.shape)

# Inner join the datasets on filename
df = pd.merge(df, image_sizes, on='filename', how='inner')

print("Combined", df.shape)

# Delete the image size df
del image_sizes

# Store checkpoint
df.to_csv(main_dir + "/image_data.csv", index=False)
#df = pd.read_csv('data/image_data.csv')

df.head()

#CORRECT BOUNDING BOXES
# Correct bounding boxes not contained by image
df.loc[df['x0'] < 0, 'x0'] = 0
df.loc[df['y0'] < 0, 'y0'] = 0
df.loc[df['x1'] > df['image_width'], 'x1'] = df['image_width']
df.loc[df['y1'] > df['image_height'], 'y1'] = df['image_height']

df.head()

# Check by replotting the image from above

# Select the dataframe row corresponding to our image
image = train_dir + "/1.png"
bbox = df[df.filename == image]
print(bbox)
# Display image
display_bbox(image, bbox)


##DROP IMAGE WITH 6 DIGITS
# Count the number of images by number of digits
df.num_digits.value_counts(sort=False)
# Keep only images with less than 6 digits
df = df[df.num_digits < 6]


##STANDARDIZE IMAGE SIZE AND CROP IMAGES
def crop_and_resize(image, img_size):
    """ Crop and resize an image
    """
    image_data = plt.imread(image['filename'])
    crop = image_data[image['y0']:image['y1'], image['x0']:image['x1'], :]
    return transform.resize(crop, img_size)

def create_dataset(df, img_size):
    """ Helper function for converting images into a numpy array
    """
    # Initialize the numpy arrays (0's are stored as 10's)
    X = np.zeros(shape=(df.shape[0], img_size[0], img_size[0], 3))
    y = np.full((df.shape[0], 5), 10, dtype=int)
    
    # Iterate over all images in the pandas dataframe (slow!)
    for i, (index, image) in enumerate(df.iterrows()):
        
        # Get the image data
        X[i] = crop_and_resize(image, img_size)
        
        # Get the label list as an array
        labels = np.array((image['labels']))
                
        # Store 0's as 0 (not 10)
        labels[labels==10] = 0
        
        # Embed labels into label array
        y[i,0:labels.shape[0]] = labels
        
    # Return data and labels   
    return X, y

# Change this to select a different image size
image_size = (32, 32)

# Get cropped images and labels (this might take a while...)
X_train, y_train = create_dataset(df[df.filename.str.contains('train')], image_size)
X_test, y_test = create_dataset(df[df.filename.str.contains('test')], image_size)
# We no longer need the dataframe
del df

print("Training", X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Helper function for plotting nrows * ncols images
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2*nrows))
    for i, ax in enumerate(axes.flat): 
        # Pretty string with actual label
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)
        if cls_pred is None:
            title = "Label: {0}".format(true_number)
        else:
            # Pretty string with predicted label
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "Label: {0}, Pred: {1}".format(true_number, pred_number)  
            
        if images[i].shape == (32, 32, 3):
            ax.imshow(images[i])
        else:
            ax.imshow(images[i,:,:,0], cmap="gray")
        #ax.imshow(images[i])
        ax.set_title(title)   
        ax.set_xticks([]); ax.set_yticks([])

# Display images from the sets
plot_images(X_train, 4, 5, y_train)
plt.show()
plot_images(X_test, 4, 5, y_test)
plt.show()

##DATA EXPLORATION
# Train set
y_train_counts = np.unique((y_train != 10).sum(1), return_counts=True)

y_train_counts = list(zip(y_train_counts[0], y_train_counts[1]))
y_train_df_counts = pd.DataFrame(y_train_counts,  columns= ['Number of Digits', 'Count'])
y_train_df_counts.set_index('Number of Digits', inplace=True)

# Test set
y_test_counts = np.unique((y_test != 10).sum(1), return_counts=True)

y_test_counts = list(zip(y_test_counts[0], y_test_counts[1]))
y_test_df_counts = pd.DataFrame(y_test_counts, columns= ['Number of Digits', 'Count'])
y_test_df_counts.set_index('Number of Digits', inplace=True)

combined_counts_df = pd.concat([y_train_df_counts, y_test_df_counts], 
                               keys=['Train', 'Test'],
                              names=['Dataset'])
combined_counts_df

# Initialize the subplots
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 4))

# Sequence length distribution - training set 
ax1.hist((y_train != 10).sum(1), bins=5)
ax1.set_title("Training set")
ax1.set_xlim(1, 5)

# Sequence length distribution - test set 
ax2.hist((y_test != 10).sum(1), bins=5, color='g')
ax2.set_title("Test set")

# Set the main figure title
fig.suptitle('Number of digits per image distribution', fontweight='bold')

plt.show()

# Initialize the subplots
fig, ax = plt.subplots(1, 2, sharex=True, figsize=(16, 4))

# Set the main figure title
fig.suptitle('Number of digits per image countplots', fontweight='bold')


sns.countplot((y_train != 10).sum(1), ax=ax[0])
ax[0].set_title("Training set")

sns.countplot((y_test != 10).sum(1), ax=ax[1])
ax[1].set_title("Testing set")

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 4))

fig.suptitle('Individial Digit Distribution', fontweight='bold')

ax1.hist(y_train.flatten(), bins=5)
ax1.set_title("Training set")

ax2.hist(y_test.flatten(), bins=5, color='g')
ax2.set_title("Test set")

plt.show()

##Create Validation Set
def random_sample(N, K):
    """Return a boolean mask of size N with K selections
    """
    mask = np.array([True]*K + [False]*(N-K))
    np.random.shuffle(mask)
    return mask

# 5% of training images for validation
sample1 = random_sample(X_train.shape[0], 2200)

# Create valdidation from the sampled data
X_val = X_train[sample1]
y_val = y_train[sample1]

# Keep the data not contained by sample
X_train = X_train[~sample1]
y_train = y_train[~sample1]

print("Training", X_train.shape, y_train.shape)
print('Validation', X_val.shape, y_val.shape)

##Store un-processed image datasets to disk
# Create file
h5f = h5py.File(main_dir + '/SVHN_multi_digit_rgb.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=X_val)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()

##IMAGE PRE-PROCESSING
#Convert to Grayscale
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

# Transform the images to greyscale
X_train = rgb2gray(X_train).astype(np.float32)
X_test = rgb2gray(X_test).astype(np.float32)
X_val = rgb2gray(X_val).astype(np.float32)

#Normalize pixels
# Calculate the mean on the training data
train_mean = np.mean(X_train, axis=0)

# Calculate the std on the training data
train_std = np.std(X_train, axis=0)

# Subtract it equally from all splits
train_norm = (X_train - train_mean) / train_std
test_norm = (X_test - train_mean)  / train_std
val_norm = (X_val - train_mean) / train_std

# Make sure that didn't break anything
plot_images(train_norm, 4, 8, y_train)
plt.show()

##Save pre-processed images
# Create file
h5f = h5py.File(main_dir + '/SVHN_multi_digit_norm_grayscale.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=train_norm)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=test_norm)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=val_norm)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()