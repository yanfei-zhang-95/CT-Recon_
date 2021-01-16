import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import cv2
import os
from data import *

working_path = "G:/project5/raw_pics/Images_png/000738_02_01/"



# 以一张切片为例
imgs = load_data(path = working_path, window = [-1024, 3071], spacing = [1, 1, 1], new_spacing = [1, 1, 1], l = 512, w = 512)
img = imgs[50]

mean = np.mean(img)
std = np.std(img)
img = img-mean
img = img/std

#提取肺部大致均值
middle = img[100:400,100:400]
mean = np.mean(middle)

# 将图片最大值和最小值替换为肺部大致均值
max = np.max(img)
min = np.min(img)
print(mean,min,max)
img[img==max]=mean
img[img==min]=mean

# 进行聚类并提取特征
kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

thresh_img = (thresh_img - np.max(thresh_img))/(np.max(thresh_img) - np.min(thresh_img))
thresh_img = (thresh_img + 1)/2

eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
dilation = morphology.dilation(eroded, np.ones([10, 10]))
labels = measure.label(dilation)
fig, ax = plt.subplots(2, 2, figsize = [8, 8])
ax[0, 0].imshow(thresh_img, cmap = 'gray')
ax[0, 1].imshow(eroded, cmap = 'gray')
ax[1, 0].imshow(dilation, cmap = 'gray')
ax[1, 1].imshow(labels, cmap = 'gray')  # 标注mask区域切片图
plt.show()

label_vals = np.unique(labels)
regions = measure.regionprops(labels) # 获取连通区域

# 设置经验值，获取肺部标签
good_labels = []
for prop in regions:
    B = prop.bbox
    print(B)
    if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
        good_labels.append(prop.label)
'''
(0L, 0L, 512L, 512L)
(190L, 253L, 409L, 384L)
(200L, 110L, 404L, 235L)
'''
# 根据肺部标签获取肺部mask，并再次进行’膨胀‘操作，以填满并扩张肺部区域
mask = np.ndarray([512,512],dtype=np.int8)
mask[:] = 0
for N in good_labels:
    mask = mask + np.where(labels==N,1,0)
mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
#imgs_to_process[i] = mask
fig,ax = plt.subplots(2,2,figsize=[10,10])
ax[0,0].imshow(img)  # CT切片图
ax[0,1].imshow(img,cmap='gray')  # CT切片灰度图
ax[1,0].imshow(mask,cmap='gray')  # 标注mask，标注区域为1，其他为0
ax[1,1].imshow(img*mask,cmap='gray')  # 标注mask区域切片图
plt.show()