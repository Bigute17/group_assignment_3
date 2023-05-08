import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import random
import os

# Set Image Directory (change this to accomodate yourself!)
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
y_train[0][0]

#Conv layers requires 4 dimensional input (i.e., if we input three dimensions and add batch dim we get 4D)
inputs = tf.keras.layers.Input(shape=(32,32,1), name='input') 
x = tf.keras.layers.Conv2D(filters=48,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(inputs)
x = tf.keras.layers.Conv2D(filters=64,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=160,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=192,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=192,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=210,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.Conv2D(filters=210,kernel_size = 3, strides = 1, padding = "same", activation = "relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = "valid")(x)
x = tf.keras.layers.BatchNormalization()(x)
#dense layers expect 1D array of features for each instance so we need to flatten.
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(3072, activation = 'relu')(x)
x = tf.keras.layers.Dense(3072, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
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

checkpoint_filepath = os.path.join(main_dir, "SVHNModel.h5")
#model = tf.keras.models.load_model(checkpoint_filepath)

#Fit model
model.fit(x=X_train,
          y=y_train,
          epochs=50,
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
        ),
        tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
        )])

model.save(checkpoint_filepath)

##Predictions
test_predictions = model.predict(X_test)
test_predictions[0][0]
print(np.argmax(test_predictions[0][0]))

for i in random.sample(range(0,10000),5):
    
    actual_labels = []
    predicted_labels = []
    plt.figure()
    plt.imshow(X_test[i])
    for j in range(0,5):
        actual_labels.append(np.argmax(y_test[j][i]))
        predicted_labels.append(np.argmax(test_predictions[j][i]))
        
    plt.title("Actual: {}\nPredicted: {}".format(actual_labels, predicted_labels))
    plt.show()
    print("Actual labels: {}".format(actual_labels))
    print("Predicted labels: {}\n".format(predicted_labels)) 
    
    
    
def calculate_acc(predictions,real_labels):
    
    individual_counter = 0
    global_sequence_counter = 0
    coverage_counter = 0
    confidence = 0.7
    for i in range(0,len(predictions[0])):
        # Reset sequence counter at the start of each image
        sequence_counter = 0 
        
        for j in range(0,5):
            
            if np.argmax(predictions[j][i]) == np.argmax(real_labels[j][i]):
                individual_counter += 1
                sequence_counter += 1
            if predictions[j][i][np.argmax(predictions[j][i])] >= confidence:
                coverage_counter += 1
        
        if sequence_counter == 5:
            global_sequence_counter += 1
         
    ind_accuracy = individual_counter / float(len(predictions[0]) * 5)
    global_accuracy = global_sequence_counter / float(len(predictions[0]))
    coverage = coverage_counter / float(len(predictions[0]) * 5)
    
    return ind_accuracy,global_accuracy, coverage

ind_acc, glob_acc, coverage = calculate_acc(test_predictions, y_test)

print("The individual accuracy is {} %".format(ind_acc * 100))
print("The sequence prediction accuracy is {} %".format(glob_acc * 100))
print("The coverage is {} %".format(coverage * 100))
