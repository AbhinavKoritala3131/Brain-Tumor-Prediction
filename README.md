# Brain Tumor Detection Using MRI Images
# Overview
This project involves building a Brain Tumor Detection System using machine learning and deep learning algorithms. The primary goal of the system is to automatically detect brain tumors from Magnetic Resonance Imaging (MRI) scans, specifically targeting Glioblastoma (GBM/HGG) and Lower Grade Glioma (LGG). We use U-Net, a convolutional neural network (CNN) model, for semantic segmentation of the MRI images to identify the presence of brain tumors.
The system leverages MRI scans in multiple modalities—T1, T1ce, T2, and FLAIR—to predict the location and segmentation of tumors, helping radiologists and doctors make quicker and more accurate decisions.

# Key Features:
-> Data Preprocessing: Image normalization, cropping, and data augmentation.
-> Model: 3D U-Net for brain tumor segmentation.
-> Prediction and Visualization: Visualization of the model's segmentation outputs and comparison with ground truth.
-> Saving Processed Data: Exporting processed images and masks as .npy files for training.

# Dataset
The dataset used in this project is from BraTS 2020 (Brain Tumor Segmentation Challenge). The dataset consists of multi-institutional, pre-operative MRI scans of High-Grade Glioma (HGG) and Lower-Grade Glioma (LGG). The images are provided in NIfTI (.nii.gz) format and include the following modalities:

->T1-weighted (T1)
->T1-contrast-enhanced (T1ce)
->T2-weighted (T2)
->Fluid Attenuated Inversion Recovery (FLAIR)
Each of the MRI scans has corresponding segmentation masks with tumor regions annotated.
# Model Architecture
This project uses U-Net, a fully convolutional network for semantic segmentation, which has the following key components:

->Contracting Path (Encoder): A series of convolutional layers followed by max-pooling layers to extract hierarchical features from the input images.
->Bottleneck: The deepest layer of the model that holds the most abstract representations.
->Expansive Path (Decoder): A series of upsampling and convolutional layers that gradually reconstruct the image to match the original size, with skip connections from the contracting path.
->Final Layer: A 3D convolution with softmax activation for pixel-wise classification (tumor vs. non-tumor regions).

# Installation

This project requires several Python libraries for data processing, model training, and evaluation. You can install all dependencies using pip

# Usage
1. Data Preprocessing:
The preprocessing script loads MRI images, scales, and normalizes them. It also prepares masks for training the model.

->import numpy as np
  import nibabel as nib
  from sklearn.preprocessing import MinMaxScaler
  from keras.utils import to_categorical


-> Load images

-> Normalize images

-> Combine images for model input

2. Model Training:
Once the data is preprocessed, you can build the U-Net model and train it using the generator.

->from keras.models import Model
  from keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, Dropout

# Define U-Net model
->def simple_unet_model(input_shape, num_classes):
      inputs = Input(input_shape)
      # Model architecture...
      return model

-> Compile and train the model

3. Model Prediction:
After training the model, you can predict tumor segmentation for test images and visualize the results.
->test_img = np.load('path/to/test_image.npy')
  test_mask = np.load('path/to/test_mask.npy')

-> Make prediction


# Visualization
import matplotlib.pyplot as plt
n_slice = 50
plt.imshow(test_prediction_argmax[:,:,n_slice])
plt.title('Predicted Tumor')
plt.show()

# Training and Testing
Training: You can train the model using the fit() function, specifying the number of epochs, batch size, and other hyperparameters.
Testing: The model's performance is evaluated by testing it on unseen images and comparing the predicted masks with the actual masks (ground truth).
Prediction
Once the model is trained, it can be used to predict tumor regions in new MRI images. The predictions can be visualized using matplotlib.

# References
BraTS 2020 dataset: BraTS 2020 Dataset
U-Net: U-Net: Convolutional Networks for Biomedical Image Segmentation
Deep Learning with Keras: Keras Documentation

# Acknowledgments
The dataset used in this project is provided by the BraTS 2020 Challenge, a leading research challenge on brain tumor segmentation.
This project is built on the work of many contributors in the field of medical imaging and deep learning.


  
