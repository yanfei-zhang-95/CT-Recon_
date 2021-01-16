import numpy as np
import pandas as pd
import os
import cv2
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation
from sklearn.cluster import KMeans
from skimage import morphology

np.set_printoptions(threshold = np.nan)

def lung_segmentation(img, n_cluster):
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[50:200, 50:200]
    mean = np.mean(middle)

    max = np.max(img)
    min = np.min(img)
    # print(mean, min, max)
    img[img == max] = mean
    img[img == min] = mean

    kmeans = KMeans(n_clusters=n_cluster).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 0.0, 1.0)  # threshold the image

    return thresh_img

def mask_refinement(learned_mask_val):

    contours, hierarchy = cv2.findContours(image = np.ndarray.astype(learned_mask_val, dtype = np.uint8), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)

    def contours_area(cnt):
        (x, y, w, h) = cv2.boundingRect(cnt)
        return w * h

    max_cnt = max(contours, key = lambda cnt: contours_area(cnt))
    mask = np.zeros_like(learned_mask_val)
    mask = cv2.drawContours(mask, [max_cnt], 0, 255, -1)
    mask_new = mask*learned_mask_val

    return mask_new

def vessel_mask_generator(img):
    [M, N] = img.shape
    new_img = img.copy()

    for i in range(N):
        start_left = new_img[0, i]
        new_img[0, i] = 1

        for j in range(1, int(M/2)+1):
            if new_img[j, i] == start_left:
                start_left = new_img[j, i]
                new_img[j, i] = 1
            else:
                break

    for i in range(N):
        start_right = new_img[-1, i]
        new_img[-1, i] = 1

        for k in range(M-2, int(M/2), -1):
            if new_img[k, i] == start_right:
                start_right = new_img[k, i]
                new_img[k, i] = 1
            else:
                break

    return new_img

def full_mask_generator(imgs, learned_mask = None):

    [N, H, W, C] = imgs.shape

    vessel_masks_val = []
    learned_masks_val = []

    for i in range(C):

        img = imgs[:, :, :, i]
        img = img.reshape([H, W])

        if learned_mask is not None:
            new_img = img.reshape([H, W]) * learned_mask.reshape([H, W])
            # new_img = np.where(new_img < -0.90, 0, new_img)

            learned_mask_val = lung_segmentation(new_img.reshape([H, W]), 2)
            learned_mask_val = -learned_mask_val + 1

        else:
            learned_mask_val = lung_segmentation(img, 2)

        learned_mask_val = mask_refinement(learned_mask_val)

        learned_mask_val = np.where(learned_mask_val == 255, 1, learned_mask_val)

        vessel_mask_val = vessel_mask_generator(learned_mask_val)

        learned_masks_val.append(learned_mask_val.reshape([N, H, W, 1]))
        vessel_masks_val.append(vessel_mask_val.reshape([N, H, W, 1]))


    learned_masks_val = np.concatenate(learned_masks_val, axis = -1)
    vessel_masks_val = np.concatenate(vessel_masks_val, axis = -1)

    vessel_masks_val = vessel_masks_val.reshape([N, H, W, C])
    learned_masks_val = learned_masks_val.reshape([N, H, W, C])

    return learned_masks_val, vessel_masks_val

def windowing(im, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)

    im1[im1 > win[1]] = win[1]
    im1[im1 < win[0]] = win[0]

    im1 = (im1 - win[0])/(win[1] - win[0])*2 - 1

    return im1

def get_mask(im):
    th = 32000 - 32768
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))

    if mask.sum() == 0:
        mask = im * 0 + 1

    return mask.astype(dtype=np.int32)

def resample(image, spacing, new_spacing):

    spacing = np.asarray(spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def resize(pic, l, w):
    pic = cv2.resize(pic, dsize = (l, w))
    return pic

def load_data(path, spacing, new_spacing, window, l, w):
    try:
        slice_name = os.listdir(path)
        slice_name.sort()

        slices = []
        #Transform slices and doing interpolation for new spacing
        for i in range(1, len(slice_name)):
            slice_num_i, _ = slice_name[i].split('.')
            slice_num_iminus1, _ = slice_name[i-1].split('.')

            if int(slice_num_i) != int(slice_num_iminus1)+1:
                break
            else:
                slices.append(cv2.imread(path+slice_name[i-1], -1).astype(np.float)-32768)

        slices = np.stack(slices)
        new_slices = resample(image = slices, spacing = spacing, new_spacing = new_spacing)

        #Adjust the pixel values to HU
        # masks = []


        windowed_slices = []
        for i in range(len(new_slices)):
            windowed_slice = resize(new_slices[i], l = l, w = w)
            # mask = lung_segmentation(windowed_slice)
            windowed_slice = windowing(windowed_slice, window)
            windowed_slices.append(windowed_slice)
            # masks.append(mask)
        windowed_slices = np.stack(windowed_slices) #resize all windowed slices from its original size to [256, 256] for training purposes

        # masks = np.stack(masks)

        # return windowed_slices, masks
        return windowed_slices

    except:
        return "Mistakes Encountered"

def getXTensors(patients, batch, type):
    tensors = []
    n = len(patients)
    n_now = n-1
    if type == 'm':#Get window from 0 to n-1, coresspond to xn1 starting from 1 to n
        patients = patients[0:n-1]
    if type == 'n1':#Get window from 1 to n, coresspond to xm starting from 0 to n-1
        patients = patients[1:n]

    for i in range(n):
        if n_now > batch:
            tensor = patients[0:batch]
            [B, H, W] = tensor.shape
            tensor = np.reshape(tensor, newshape = [1, B, H, W])
            tensors.append(tensor)
            n_now = len(patients)
            patients = patients[1:n_now]
        else:
            break

    return np.stack(tensors)

def getYTensors(patients, batch, type):
    [B, H, W] = patients.shape

    if type == 'm':
        y_tensor = patients[0: B-batch]
    if type == 'n1':
        y_tensor = patients[batch: B]
    y_tensor = np.reshape(y_tensor, newshape = [B-batch, 1, H, W])

    return y_tensor

def get_data(patient, block):
    Xm_tensor = getXTensors(patients=patient, batch=block, type='m')  # get N-block-1 (1, block, 256, 256, 1) 5-dimensional tensors
    Xn1_tensor = getXTensors(patients=patient, batch=block, type='n1')  # get N-block-1 (1, block, 256, 256, 1) 5-dimensional tensors
    Ym_tensor = getYTensors(patients=patient, batch=block,  type='m')  # get a (N-block, 256, 256, 1) 4-dimensional tensor
    Yn1_tensor = getYTensors(patients=patient, batch=block, type='n1')  # get a (N-block, 256, 256, 1) 4-dimensional tensor

    data = [Xm_tensor, Xn1_tensor, Ym_tensor, Yn1_tensor]
    len = Yn1_tensor.__len__()
    return data, len

def get_single_data(data, num, batch, block, shape, chn):
    Xm_tensor, Xn1_tensor, Ym_tensor, Yn1_tensor= data

    def reshapeX(tensorX):
        reshapedTensor = []
        for i in range(block):
            reshapedTensor.append(tensorX[:,i,:,:])
        return np.stack(reshapedTensor, axis = -1)

    batch_Xm = reshapeX(Xm_tensor[num])
    batch_Ym = Ym_tensor[num].reshape([batch, shape, shape, chn])

    batch_Xn1 = reshapeX(Xn1_tensor[num])
    batch_Yn1 = Yn1_tensor[num].reshape([batch, shape, shape, chn])

    data = [batch_Xm, batch_Ym, batch_Xn1, batch_Yn1]
    return data, num