import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Activation, Convolution2D, Flatten, MaxPooling2D, Dropout, SpatialDropout2D, Lambda
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

##
# Global vars
#
debug = True
restore = 'restore.hdf5'
steering_range = 0.1
zero_prob = 0.0
split_size = 0.9

##
# Sanity check: if restoring, warning
#
if os.path.isfile(restore):
    print('#############################\n#  WARNING: MODEL RESTORE   #\n#############################')


##
# Read the recorded data and create a shuffled dataset
#
rec_dir = 'data'
recording_dirs = [os.path.join(rec_dir, name) for name in os.listdir(rec_dir) if os.path.isdir(os.path.join(rec_dir, name)) and ('recording' in name and not 'not' in name)]

header = ['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']

header_dict = {'center_image': 0,
               'left_image': 1,
               'right_image': 2,
               'steering_angle': 3,
               'throttle': 4,
               'break': 5,
               'speed': 6}

dataset = pd.DataFrame()

# Process every directory and read the data in it
for d in recording_dirs:
    print('Processing {}'.format(d))
    tmp = pd.read_csv(os.path.join(d, 'driving_log.csv'), header=None, names=header, comment='#')
    tmp.left_image = d + os.path.sep + tmp.left_image.str.strip()
    tmp.center_image = d + os.path.sep + tmp.center_image.str.strip()
    tmp.right_image = d + os.path.sep + tmp.right_image.str.strip()
    dataset = pd.concat([dataset, tmp])

# Reset index to avoid duplicated index (different files starts with 0 index)
dataset.reset_index(inplace=True, drop=True)


##
# Filter unwanted data
#

# Low speed (initial images, crashes, etc)
dataset = dataset[dataset.speed > 25]

# Hard breaking (dangerous situations, going off the road, etc.)
dataset = dataset[dataset['break'] < 0.1]


##
# Define a functoin to shuffle the dataset
#
def shuffle_ds(dataset):
    # Shuffle data to avoid training on same series of recordings
    dataset = dataset.reindex(np.random.permutation(dataset.index))

    # Reset index again to make it sequential
    dataset.reset_index(inplace=True, drop=True)

    return dataset


# Shuffle the dataset
dataset = shuffle_ds(dataset)
# Save the shuffled file to feed it into model.fit_generator
dataset_train_path = '/tmp/behavioral-cloning_train.csv'
dataset_val_path = '/tmp/behavioral-cloning_validation.csv'
# Split the sata for training and validation
# Testing is done on the road :-)
split = int(len(dataset) * split_size)
dataset[:split].to_csv(dataset_train_path, header=False, index=False)
dataset[split:].to_csv(dataset_val_path, header=False, index=False)

print('Dataset created')


def model_lenet(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name='image_normalization', input_shape=input_shape))

    model.add(Convolution2D(6, 5, 5, name='convolution_1', subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Convolution2D(16, 5, 5, name='convolution_2', subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(Dense(120, name='hidden1'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(84, name='hidden2'))
    model.add(ELU())
    model.add(Dense(10, name='hidden3'))
    model.add(ELU())

    model.add(Dense(1, name='steering_angle'))

    return model


##
# Data generator
#

##
# read_image()
#
# input: path to image
# output: image preprocessed
#
def read_image(path, resize_shape = (32, 32)):
    # Read the image
    img = cv2.imread(path)
    # Crop the road area (no sky, no car)
    low_l = int(0.3125 * img.shape[0])
    sup_l = int(0.75 * img.shape[0])
    img = img[low_l:sup_l,:]
    # Resize the image
    img = cv2.resize(img, resize_shape)
    # Transform to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

##
# process_line()
#
# input: line of csv (from dataset)
# output: one image with its label
8
def process_line(line, augment = False, resize_shape=(32, 32)):
    row = line.split(',')

    steering = row[header_dict['steering_angle']]
    steering = float(steering)

    steering_angles = []
    imgs = []

    # Sample images proporionally to their angles (with a minimum of min_chance% chance)
    min_chance = 0.05
    if np.random.rand() < ((min_chance + abs(steering)) / (1 + min_chance)):
        img = read_image(row[header_dict['center_image']], resize_shape)
        # Add random noise to prevent overfitting
        steering += np.random.rand() * 1e-4

        if np.random.rand() < 0.5:
            # Center image
            steering_angles.append(steering)
            imgs.append(img)
        else:
            # Center flipped image
            img = cv2.flip(img, 1)
            imgs.append(img)
            steering_angles.append(-steering)

    steering_angles = np.array(steering_angles)
    imgs = np.array(imgs)

    return imgs, steering_angles

##
# get_data()
#
# input: path to dataset, batch_size
# output: batch of images with labels
#
def get_data(path, batch_size = 64, augment = False, resize_shape=(32, 32), output_shape=(-1, 32, 32, 3)):

    while True:
        features = None
        labels = None
        batch_count = 0

        with open(path) as f:
            for line in f:
                X, y = process_line(line, augment, resize_shape=resize_shape)

                # A line of data could generate more than 1 instances
                n_data = len(y)
                if n_data > 0:
                    # Initialization
                    if features is None:
                        features = X
                    else:
                        features = np.concatenate((features, X))

                    if labels is None:
                        labels = y
                    else:
                        labels = np.concatenate((labels, y))

                    # Start processing the data
                    batch_count += n_data

                    if batch_count > batch_size:
                        # Where to cut the batch: is it batch_size or less?
                        batch_cut = min(batch_size, batch_count)

                        features = np.reshape(features, output_shape)
                        labels = np.reshape(labels, (batch_count, -1))

                        # Return batch_size or less
                        r_features = features[0:batch_cut]
                        r_labels = labels[0:batch_cut]

                        # Save the excedent to next iteration or nullify the variables to
                        # start fresh
                        batch_count = batch_size - batch_cut
                        if batch_count > 0:
                            features = list(features[batch_cut:batch_cut + batch_size])
                            labels = list(labels[batch_cut:batch_cut + batch_size])
                        else:
                            features = None
                            labels = None

                        yield (r_features, r_labels)
print('Generator created')


##
# get_model()
#
# Return a previously saved model or a new one with corresponding input sizes and generators
#
# input: model name
# output: model, training generator, validation generator
#
def get_model(batch_size=64):

    if os.path.isfile(restore):
        model = load_model(restore)
        layer_0 = model.layers[0].get_config()
        resize_shape = (layer_0['batch_input_shape'][2], layer_0['batch_input_shape'][1])
        input_shape = layer_0['batch_input_shape'][1:]
        output_shape = (-1,) + input_shape
        print('Model restored')
    else:
        resize_shape = (32, 32)
        input_shape = (32, 32, 1)
        model = model_lenet(input_shape)

        output_shape = (-1,) + input_shape
        print('Model compiled')
    
    training_generator = get_data(dataset_train_path, batch_size=batch_size, augment=True, resize_shape=resize_shape, output_shape=output_shape)
    validation_generator = get_data(dataset_val_path, batch_size=1, augment=False, resize_shape=resize_shape, output_shape=output_shape)
        
    return model, training_generator, validation_generator


##
# Start training
#
batch_size = 64
samples_per_epoch = int(split / batch_size) * batch_size * 5
samples_per_val = len(dataset) - split
nb_epoch = 20

# Model and generators
model_name = 'lenet'
model, training_generator, validation_generator = get_model(batch_size=batch_size)

# Optimizer
optimizer = Adam(1e-4)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Checkpoint
save_dir = datetime.utcnow().strftime('%d-%m-%Y_%H_%M_%S')
save_dir = os.path.join('saved', save_dir)
os.mkdir(save_dir)
filename = model_name + '-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5'
filepath = os.path.join(save_dir, filename)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', verbose=0, patience=3, mode='min')

callbacks_list = [checkpoint]

# Train the model
history = model.fit_generator(
  training_generator,
  samples_per_epoch = samples_per_epoch,
  nb_epoch = nb_epoch,
  validation_data = validation_generator,
  nb_val_samples = samples_per_val,
  callbacks = callbacks_list
  )

print('Model trained')
