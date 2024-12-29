# This app generates symptoms and age, and assigns a value 0 or 1 that represents if the sample is pneumonia or not
# after that it loads the symptoms, age and labels appropriatly in np array form,
# and finally it splits the data randomly into 2 catagorys
# 70% training 30% not
# it then splits the 30% a second: 50% testing and 50% validation
# Which gives as a final distribution like so: [0.7 training , 0.15 testing , 0.15 validation]

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# two lists that define what pnemonia symptoms are
PNEUMONIA_SYMPTOMS = ["fever", "cough", "shortness of breath", "chills", "fatigue", "chest pain", "high temperature", "tiredness", "wheezing"]
OTHER_SYMPTOMS = ["headache", "nausea", "sore throat", "dizziness", "runny nose"]

MINIMUM_AGE = 18
MAXIMUM_AGE = 90

# combines both lists for the feature vect
ALL_SYMPTOMS = PNEUMONIA_SYMPTOMS + OTHER_SYMPTOMS

# number pf samples to generate
NUM_SAMPLES = 1000000

# function to generate dataset
def generate_pneumonia_data(num_samples):
    data = []
    for _ in range(num_samples):
        # assign a random age from 18 to 90
        age = random.randint(MINIMUM_AGE, MAXIMUM_AGE)
        # choose if a sample has pneumonia or not at random
        if age > 65: # if patient is old
            # assign a random number between 0 and 1
            # with a 30% of it being 0
            # and 70% change of it being 1
            has_pneumonia = random.choices([0, 1], weights=(0.3, 0.7))
        else: # if patient is not old
            # same but flipped chances
            has_pneumonia = random.choices([0, 1], weights=(0.7, 0.3))
        if has_pneumonia: # if it has pneumonia
            # generate sample from the pneumonia list of size 2 to 9
            symptom_sample = random.sample(PNEUMONIA_SYMPTOMS, random.randint(2, len(PNEUMONIA_SYMPTOMS)))
        else: # if it doesn't have pneumonia
            # generate sample from the pneumonia list of size 2 to 5
            symptom_sample = random.sample(OTHER_SYMPTOMS, random.randint(2, len(OTHER_SYMPTOMS)))
        
        # create a feature vector based on the symptom set
        # if i is in symptom sample, 1
        # otherwise, 0
        feature_vector = [1 if i in symptom_sample else 0 for i in ALL_SYMPTOMS]
        
        # append to data as a dictionary
        data.append({"symptoms": symptom_sample, "label": has_pneumonia, "age": age, "features": feature_vector})
    
    return pd.DataFrame(data) # return it in the form of a pandas DataFrame

# generate the dataset
df = generate_pneumonia_data(NUM_SAMPLES) # calls the method and returns the data to df
print("Sample synthetic data:")
print(df.head()) # prints the first 5 elements of df

# extract feature vectors and labels
X = np.array([list(row["features"]) + [row["age"]] for _, row in df.iterrows()]) # feature vectors + the age
y = df["label"].values # the labels

# split dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# print dataset shapes
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# saves the processed datasets into .npy files
np.save("X_train2.npy", X_train)
np.save("y_train2.npy", y_train)
np.save("X_val2.npy", X_val)
np.save("y_val2.npy", y_val)
np.save("X_test2.npy", X_test)
np.save("y_test2.npy", y_test)

print("Synthetic data generation and preprocessing complete.")
