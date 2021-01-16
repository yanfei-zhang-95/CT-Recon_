import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import cv2

real_img = np.load('real_img.npy')
vessel_mask_pred = np.load('vessel_mask_pred.npy')
learned_mask_pred = np.load('learned_mask_pred.npy')

learned_mask_pred = learned_mask_pred.reshape([1, 256, 256, 1])
vessel_mask_pred = vessel_mask_pred.reshape([1, 256, 256, 1])
real_img = real_img.reshape([1, 256, 256, 1])

learned_mask, vessel_mask = data.full_mask_generator(imgs = real_img, learned_mask = learned_mask_pred, vessel_mask = vessel_mask_pred)