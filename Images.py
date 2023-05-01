import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import random
import os

# Set Image Directory (change this to accomodate yourself)
main_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'BZAN554', 'Group_Assignments', 'group_assignment_3', 'SVHN')
train_dir = os.path.join(main_dir, 'train')
test_dir = os.path.join(main_dir, 'test')

##Load pre-processed data
# Open the HDF5 file containing the datasets
with h5py.File(os.path.join(main_dir, 'SVHN_multi_digit_norm_grayscale.h5'),'r') as h5f:
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

## Plot images
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

plot_images(X_train, 2, 4, y_train)
plt.show()


possible_classes = 11
def convert_labels(labels):
    
    # As per Keras conventions, the multiple labels need to be of the form [array_digit1,...5]
    # Each digit array will be of shape (60000,11)
        
    # Declare output ndarrays
    # 5 for digits, 11 for possible classes  
    dig0_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig1_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig2_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig3_arr = np.ndarray(shape=(len(labels),possible_classes))
    dig4_arr = np.ndarray(shape=(len(labels),possible_classes))
    
    for index,label in enumerate(labels):
        
        # Using np_utils from keras to OHE the labels in the image
        dig0_arr[index,:] = np_utils.to_categorical(label[0],possible_classes)
        dig1_arr[index,:] = np_utils.to_categorical(label[1],possible_classes)
        dig2_arr[index,:] = np_utils.to_categorical(label[2],possible_classes)
        dig3_arr[index,:] = np_utils.to_categorical(label[3],possible_classes)
        dig4_arr[index,:] = np_utils.to_categorical(label[4],possible_classes)
        
    return [dig0_arr,dig1_arr,dig2_arr,dig3_arr,dig4_arr]

y_train = convert_labels(y_train)
y_test = convert_labels(y_test)
y_val = convert_labels(y_val)

np.shape(y_train[0])


#Conv layers requires 4 dimensional input (i.e., if we input three dimensions and add batch dim we get 4D)
inputs = tf.keras.layers.Input(shape=(32,32,1), name='input') 
#Conv2D layer (2D does not refer to gray scale (a PET scan would be 3D))
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 7, strides = 1, padding = "same", activation = "relu")(inputs)
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
yhat1 = tf.keras.layers.Dense(11, activation = 'softmax', name = "yhat1")(x)
yhat2 = tf.keras.layers.Dense(11, activation = 'softmax', name = "yhat2")(x)
yhat3 = tf.keras.layers.Dense(11, activation = 'softmax', name = "yhat3")(x)
yhat4 = tf.keras.layers.Dense(11, activation = 'softmax', name = "yhat4")(x)
yhat5 = tf.keras.layers.Dense(11, activation = 'softmax', name = "yhat5")(x)


#Why do we stack two convolutional layers followed by a pooling layer, as opposed to having each convolutional layer followed by a pooling layer?
# Answer: every convolutional layer creates a number of feature maps (e.g,64) that are individually connected to the previous layer.
# By stacking two convolutional layers before inserting a pooling layer we allow the second convolutional layer to learn from the noisy signal, as opposed to the clean signal.

model = tf.keras.Model(inputs = inputs, outputs = [yhat1,yhat2,yhat3,yhat4,yhat5])
model.summary()


#Compile model
model.compile(loss = "categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

#Fit model
model.fit(x=X_train,
          y=y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_val, y_val),
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
    ])

##Predictions
test_predictions = model.predict(X_test)
test_predictions[0][0]
print(np.argmax(test_predictions[0][0]))


for i in random.sample(range(0,10000),5):
    
    actual_labels = []
    predicted_labels = []
    plt.figure()
    plt.imshow(X_test[i])
    plt.show()
    for j in range(0,5):
        actual_labels.append(np.argmax(y_test[j][i]))
        predicted_labels.append(np.argmax(test_predictions[j][i]))
        
    print("Actual labels: {}".format(actual_labels))
    print("Predicted labels: {}\n".format(predicted_labels))