import tensorflow as tf
import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import model
import json

from sklearn.cluster import KMeans

def info_acquisition(INFO_PATH, type):

    if type == 'Training':
        mode = 1
    elif type == 'Testing':
        mode = 2
    else:
        mode = 3

    dl_info = pd.read_csv(INFO_PATH)

    dl_info.drop_duplicates(subset = ['Patient_index', 'Study_index', 'Series_ID', 'Slice_range'], keep = 'first', inplace = True)
    dl_info.drop(dl_info[dl_info['Train_Val_Test'] != mode].index)

    df1 = dl_info['File_name'].str.split('_', expand = True)
    df1['Folder'] = df1[0].str.cat([df1[1], df1[2]], sep = '_')
    df1['Spacing_mm_px_'] = dl_info['Spacing_mm_px_']
    df1 = df1.drop(columns = [0, 1, 2, 3])
    df1['Slice_range'] = dl_info['Slice_range']

    df2 = df1['Spacing_mm_px_'].str.split(', ', expand = True)
    df1 = pd.merge(df1, df2, left_index = True, right_index = True)

    df1 = df1.drop(columns = ['Spacing_mm_px_'])
    df1.reset_index(drop = True, inplace = True)
    df1 = df1.rename(columns = {0: "Horizontal", 1: "Vertical", 2: "Distances"})
    df1[['Horizontal', 'Vertical', 'Distances']] = df1[['Horizontal', 'Vertical', 'Distances']].apply(pd.to_numeric)

    df1 = df1.drop(df1[df1['Distances'] != 1].index)
    df1 = df1.drop(columns = ['Horizontal', 'Vertical', 'Distances'])
    df1.reset_index(drop = True, inplace = True)

    return df1


def list_generator(df1, block, new_spacing):
    total_list = []

    for i in range(len(df1)):

        folder = df1['Folder'][i]
        slice_range = df1['Slice_range'][i]
        slice_range = slice_range.split(",")

        current_range = np.arange(start = int(slice_range[0]), stop = int(slice_range[1]), step = new_spacing)

        for j in range(len(current_range) - block - 1):
            start = j
            end = j + block + 1
            indices = current_range[start:end]
            indices = list(map(lambda x: str(x), indices))
            indices = list(map(lambda x: folder + '/00' + x + '.png' if len(x) == 1 else (folder + '/0' + x + '.png' if len(x) == 2
                                                                                          else folder + '/' + x + '.png'), indices))
            total_list.append(indices)

    return total_list

def windowing(im, win, mode = 'Init'):
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)

    if mode == 'Init':
        im1[im1 > win[1]] = win[1]
        im1[im1 < win[0]] = win[0]
    elif mode == 'Postprocessing':
        im1[im1 > win[1]] = 1/2*(win[1] - win[0]) + win[0]
        im1[im1 < win[0]] = 1/2*(win[1] - win[0]) + win[0]

    im1 = (im1 - win[0]) / (win[1] - win[0]) * 2 - 1

    return im1

def slice_generator(current_list, shape, PATIENT_PATH, WINDOW):
    slices = []
    for pic_path in current_list:
        try:
            picture = cv2.imread(PATIENT_PATH + pic_path, -1).astype(np.float) - 32768
            picture = cv2.resize(picture, dsize = (shape, shape))
            picture = windowing(picture, win = WINDOW)

            slices.append(picture)
        except:
            return



    X = slices[0:-1]
    X = np.stack(X, axis = -1)
    [H, W, C] = X.shape
    X = X.reshape([1, H, W, C])

    y = slices[-1]
    y = y.reshape([1, H, W, 1])

    return X, y

def lung_segmentation(img, n_cluster):
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[50:200, 50:200]
    mean = np.mean(middle)

    max = np.max(img)
    min = np.min(img)
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
    mask_new = mask * learned_mask_val

    return mask_new

def vessel_mask_generator(img):
    [H, W] = img.shape
    new_img = img.copy()

    for i in range(W):
        start_left = new_img[0, i]
        new_img[0, i] = 1

        for j in range(1, int(H/2)+1):
            if new_img[j, i] == start_left:
                start_left = new_img[j, i]
                new_img[j, i] = 1
            else:
                break

    for i in range(W):
        start_right = new_img[-1, i]
        new_img[-1, i] = 1

        for k in range(H-2, int(H/2), -1):
            if new_img[k, i] == start_right:
                start_right = new_img[k, i]
                new_img[k, i] = 1
            else:
                break

    return new_img

def maskGenerator(img):

    [N, H, W, C] = img.shape
    img = img.reshape([H, W])

    raw_mask = lung_segmentation(img, n_cluster = 2)
    refined_raw_mask = mask_refinement(raw_mask)
    body_mask = np.where(refined_raw_mask == 255, 1, refined_raw_mask)
    vessel_mask = vessel_mask_generator(body_mask)

    return body_mask, vessel_mask

def float_range(start, stop, step):
    return [start+i*step for i in range(int((stop-start)//step))]

def json_read_write(file_name, mode, target = None):
    if mode == 'save':
        with open(file_name, 'w') as file_object:
            json.dump(target, file_object)
        print('.json file saved')
    elif mode == 'read':
        with open(file_name, 'r') as file_object:
            contents = json.load(file_object)
        return contents


if __name__ == '__main__':

    INFO_PATH = 'F:/project5/raw_pics/DL_info.csv'
    PATIENT_PATH = 'F:/project5/raw_pics/Images_png/'
    WINDOW = [-1024, 3071]
    block = 4
    shape = 256
    new_spacing = 4
    batch = 1
    chn = 1


    # Training Data
    df_training = info_acquisition(INFO_PATH, 'Training')
    training_list = list_generator(df_training, block = block, new_spacing = new_spacing)
    CKPT_BASELINE_FOLDER = "F:/project5/ckpt/baseline_ckpt/"

    random_training_list = training_list.copy()
    # random_training_list = random_training_list[0:100]
    random.shuffle(random_training_list)  # Shuffle the training procedure to make a better and universal learning


    # Define placeholders
    X = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, block * chn])
    y = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    vessel_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    body_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    y_basic_plac = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    placeholders = [X, y, vessel_mask, body_mask, y_basic_plac]

    y_basic, basic_var = model.modelStructure.basic_model(X)

    # Initialise Training Process
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Load the pre-trained baseline model
    saver_pre = tf.train.Saver(basic_var)
    saver_pre.restore(sess, CKPT_BASELINE_FOLDER + 'model.ckpt')

    bodys = []
    vessels = []

    for i in range(len(training_list)):

        data = slice_generator(random_training_list[i], shape = shape, PATIENT_PATH = PATIENT_PATH, WINDOW = WINDOW)

        if data is not None:
            X_train, y_train = data
        else:
            print('Acquired nothing.')
            continue



        y_basic_val = sess.run(y_basic, feed_dict = {X: X_train})

        body_mask, vessel_mask = maskGenerator(img = y_basic_val)

        body = y_train.reshape([shape, shape])*body_mask
        vessel = y_train.reshape([shape, shape])*(-vessel_mask + 1)

        # Rescaled Body and Vessel
        vessel = (vessel + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
        body = (body + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]

        max_vessel = 75
        min_vessel = -1000

        max_body = 720
        min_body = -360

        vessel = windowing(vessel, [min_vessel, max_vessel])
        body = windowing(body, [min_body, max_body])

        vessel = vessel*(1-vessel_mask)
        body = body*body_mask

        body = np.where(body_mask == 0, 255, body)
        vessel = np.where(1-vessel_mask == 0, 255, vessel)

        body = np.reshape(body, [shape*shape*chn*batch, 1])
        vessel = np.reshape(vessel, [shape*shape*chn*batch, 1])

        for pix_val_b in body:
            if pix_val_b != 255:
                bodys.append(pix_val_b)

        for pix_val_v in vessel:
            if pix_val_v != 255:
                vessels.append(pix_val_v)

        if i % 100 == 0:
            print('%i finished'%i)