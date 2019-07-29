from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from numpy import array, newaxis, expand_dims

def call_image(path):
    img=cv2.imread(path, cv2.IMREAD_COLOR)[...,::-1]
    return img

def split_into_rgb_channels(image):
    """Split the target image into its red, green and blue channels.image - a numpy array of shape (rows, columns, 3).
    output - three numpy arrays of shape (rows, columns) and dtype same as
    image, containing the corresponding channels.
    """
    red = image[:,:,2]
    green = image[:,:,1]
    blue = image[:,:,0]
    return red, green, blue

def denoise(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    img=norm_image
    r,g,b=split_into_rgb_channels(img)
    gaussian = cv2.GaussianBlur(b,(3,3),0)
    return gaussian

def array_resize(img):
    array=img_to_array(img)
    array = resize(array, (128, 128, 1), mode = 'constant', 
                  preserve_range = True)
    array=array/255.0
    array=array[newaxis,:,:,:]
    return array

def prediction(model, img):
    pred=model.predict(img)
    preds_t=(pred>0.5).astype(np.uint8)
    return pred, preds_t

import scipy.ndimage as ndimage
def mask_w_predicted(pred):
    orig_mask=pred.squeeze()
    dilated=ndimage.binary_dilation(orig_mask,structure=np.ones((5,5))).astype(orig_mask.dtype)
    eroded= ndimage.binary_erosion(dilated).astype(orig_mask.dtype)
    mask=eroded
    return mask

def rotate_bound(image, angle,cX,cY):
    # grab the dimensions of the image
    (h, w) = image.shape[:2] 
    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_mag_ang(img):

    """
    Gets image gradient (magnitude) and orientation (angle)

    Args:
        img

    Returns:
        Gradient, orientation
    """

    img = np.sqrt(img)

    gx = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(np.float32(img), cv2.CV_32F, 0, 1)

    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang, gx, gy 

def inital_processing(name):
    img=call_image(name)
    denoised_img=denoise(img)
    resized=array_resize(denoised_img)
    return(resized)

def mse(imageA, imageB,contour_size):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(contour_size)
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err