# Capstone-2

Skin melanoma has increased over the last decades. Although easy to treat if found early in its development, 5 year survival rate drops sharply if found after it spread to lymph node. Therefore, an early detection of the skin melanoma is vital for patient's survival. However as early melanoma can often be mistaken for harmless mole, not every patients will go straightly to a doctor to get a diagnosis. Using the previous data of malignant and benign melanoma, I made the skin melanoma detection system using openCV and Keras library.

1. Data wrangling

Capstone 2 - Downloading data and EDA using meta data

Downloads the ISIC data from the website and save images to pwd/Data/Images and meta data to pwd/Data/Descriptions.
Identify benign/malignant using metadata to be used for later when training models

2. Image processing

- function.py contains many of the functions that are used in the 

2-1 Testing on single image

Capstone 2 - Single image segmentation Traditional image preprocessing

Attempts image segmentation using image normalization, color segmentation, gaussian blur, and thresholding using edge detection.
Erosion and dilation performed to remove noise. 
The largest contour that does not have a flat line that covers more than 60% of either horizontal or vertical edge is considered to be the lesion.
The segmented image is pickled into 'testimage' to be analyzed in the next step.

Capstone 2 - Single image Traditional image processing

'testimage' is used to analyze the characteristic of the segmented image.
Following the ABCDE rule, the segmented image was checked for Asymmetry, Border, and Color.
Diameter and evolution was not used as those were not something that could be identified using the given image.
Asymmetry: measures how different each pixels are when the image in quetion was flipped horizontally or vertically.
Border: measures the total gradient of the edges of the lesion
Color: calculates the standard deviation of Red Green and Blue histogram. 

2-2 Testing on multiple images

Capstone 2 - Multiple images segmentation and image processing - Decision Tree, Random Forest

Attempts to use previously tested method from 2-1 on all the images in the folder.
After each features are converted into respective numeric values, decision tree and random forest classifiers were used.
The resulting model implies that in this case random forest classifier perform better than decision tree classifier due to its ROC value

3. Data processing using trained model

3-1. Build image segmentation model

Capstone 2 - Building UNET model for image segmentation

The algorithm used in the model for image segmentation is UNET which is a model specifically built for biomedical application.
In order to train this model, mask images used in ISIC 2018 challenge were used.
Code used for making the model was adapted from towardsdatascience website. 
The script will generate a model fitted to the given data set that will be used in the following procedures.

Capstone 2 - Multiple image segmentation using UNET model

Apply UNET model to all the images to identify skin lesion.
Images were resized to facilitate the process and analyzed for symmetry, border, and color.
The values were saved as dictionaries 

3-2. Data processing with UNET

Capstone 2- Malignancy identification over multiple images using UNET segmentation

Dictionaries acquired from UNET segmentation were used in Decision Tree Classifier and Random Forest Classifier

Capstone 2 - Transfer learning

To increase the accuracy by letting the algorithm identify trainable features, another algorithm was implemented for classification.
The algorithm used for classification is Inception V3 and was acquired from tensorflow guide webpage(https://www.tensorflow.org/hub/tutorials/image_retraining)
