# This app imports the dataest from HuggerFace, 
# then applies the preprocessing method, 
# after that it loads the images and labels appropriatly in np array form,
# and finally it splits the data randomly into 2 catagorys
# 70% training 30% not
# it then splits the 30% a second: 50% testing and 50% validation
# Which gives as a final distribution like so [0.7 training , 0.15 testing , 0.15 validation]

from datasets import load_dataset # the dataset
import numpy as np # for array manipulation and saving/storing files
import tensorflow as tf # used in preprocessing the images
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator # used in applying random transformations on images for generalization
from sklearn.model_selection import train_test_split # for splitting the dataset into the aformentioned categories
import os # to check the filepath (not used much in this app)
from collections import Counter # to check some distrubtions (not used much in this app)


IMAGE_SIZE = (640, 640)  # 640x640 pixels
BATCH_SIZE = 32 # size of the batch

# this loads the dataset with full configuration
dataset = load_dataset("keremberke/chest-xray-classification",'full', split="train")

# this function preprocesses data
def preprocess_entry(entry):
    image_path = entry['image_file_path']
    
    if not os.path.exists(image_path): # checks if the file exists (this currently does not work properly (dont worry about it :]))
        print(f"Skipping missing file: {image_path}")
        return None 

    try:
        image = tf.image.resize(entry['image'], IMAGE_SIZE)  # makes sure the image follows the 640x640 structure
        image = tf.cast(image, tf.float32) / 255.0  # normalize the image so that pixel values range from 0 to 1
        if image.shape[-1] == 1:
            image = tf.expand_dims(image, axis=-1)  # add channel dimension for grayscale
        return {"image": image, "labels": entry["labels"]} # returns a dict containing the image and it's label
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None  # Skip this image if there's an error



# calls the function
processed_data = dataset.map(lambda entry: preprocess_entry(entry), remove_columns=["image_file_path"])


#saves the images and labels in easy np arrays
images = np.array([tf.convert_to_tensor(entry['image']).numpy() for entry in processed_data])
labels = np.array([entry['labels'] for entry in processed_data])



# splits the dataset into the 3 categories (testing, training, validation)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


# data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE)

# print dataset shapes
print("Training set: {X_train.shape}, {y_train.shape}")
print("Validation set: {X_val.shape}, {y_val.shape}")
print("Test set: {X_test.shape}, {y_test.shape}")



# The follwing code has been commented out.
# It saves every category into a file, the X represents images, y represents labels.
# Run it if you find corruption in the provided files.
# WARNING: running the code bellow WILL overwrite all .npy files

'''
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("SAVED")
'''
