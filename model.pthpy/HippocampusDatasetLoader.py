"""This module defines a function to load and preprocess the hippocampus dataset, making it ready for model training.

LoadHippocampusData function:

This function loads the hippocampus dataset, which consists of 3D medical image volumes and their corresponding labels (segmentation masks).
It takes as input the root directory containing the dataset, as well as the desired output shape for the images.
The dataset is loaded from two directories: images for the image volumes and labels for the corresponding segmentation labels.
The function iterates over all image files, loading both the image and its label using the MedPy load function.
The images are normalized to the [0, 1] range by subtracting the minimum value and dividing by the range (max - min).
Both the images and labels are reshaped to a consistent size using the med_reshape function, ensuring they are compatible with the CNN input requirements.
The reshaped data is stored in a dictionary containing the image, segmentation mask, and the filename, which is then appended to a list.
med_reshape function:

This helper function reshapes a 3D image to a new size by padding it with zeros, leaving the original content in the top-left corner.
The new shape is specified as a 3-tuple, and the function ensures that no content is discarded during reshaping, with excess space padded with zeros.
Output:

The function returns a NumPy array of dictionaries, each containing the reshaped image and label data, making the dataset ready for model training.
The total number of processed slices and files is printed as a summary.
The dataset is assumed to fit into memory (about 300MB), so it is fully loaded into RAM for fast access during training.

NOTE: The dataset loading process can be optimized further for larger datasets that do not fit into memory by using techniques like memory-mapped files.

"""
"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

#from utils.utils import med_reshape

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header
        # since we will not use it
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: Les labels représentent des classes discrètes (0, 1, 2...), donc on doit les convertir
        # en entiers pour qu’ils soient compatibles avec les fonctions de perte comme CrossEntropyLoss.

        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape

    Returns:
        3D array of desired shape, padded with zeroes
    """

    reshaped_image = np.zeros(new_shape)

    # TASK: write your original image into the reshaped image
    x_max = min(image.shape[0], new_shape[0])
    y_max = min(image.shape[1], new_shape[1])
    z_max = min(image.shape[2], new_shape[2])

    reshaped_image[:x_max, :y_max, :z_max] = image[:x_max, :y_max, :z_max]

    return reshaped_image

  
