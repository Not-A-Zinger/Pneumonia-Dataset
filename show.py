# This program simply reads the pixels in a given image and displays it using pyploy
# Only used as confirmation that the data is valid and correctly represents actual images

import numpy as np
import matplotlib.pyplot as plt # the library that plots pixel values into real images

# loads file
X_train = np.load('X_train.npy')

# debugging info
print(f"X_train shape: {X_train.shape}")

# set the numbers of displayed images (I recommend 10)
num_images_to_display = 10

# creates the figure that will be drawn on
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

# for every image (currently 10)
for i in range(num_images_to_display):
    # If the image is grayscale, the shape will be (height, width, 1), and for RGB, (height, width, 3)
    image = X_train[i]
    
    # if the image has 1 channel (grayscale)
    if image.shape[-1] == 1:
        image = np.squeeze(image, axis=-1)  # Convert to 2D for grayscale images
        
    axes[i].imshow(image)
    axes[i].axis('off') 
    axes[i].set_title(f"Image {i+1}")

plt.tight_layout()
plt.show()
