import os
import cv2
import skimage

import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

import model
import data

def create_dir(dirName):
    try:
        os.mkdir(dirName)
        # print("Directory ", dirName, " Created")
    except FileExistsError:
        return

def info_acquisition(INFO_PATH):
    # Acquire information
    dl_info = pd.read_csv(INFO_PATH)

    df1 = dl_info['File_name'].str.split('_', expand = True)
    df1['Folder'] = df1[0].str.cat([df1[1], df1[2]], sep = '_')

    df1['Key_slice_index'] = dl_info['Key_slice_index']
    df1['Spacing_mm_px_'] = dl_info['Spacing_mm_px_']
    df1 = df1.drop(columns = [0, 1, 2, 3])

    df2 = df1['Spacing_mm_px_'].str.split(', ', expand = True)
    df1 = pd.merge(df1, df2, left_index = True, right_index = True)

    df1 = df1.drop(columns = ['Spacing_mm_px_'])
    df1.reset_index(drop = True, inplace = True)
    df1 = df1.rename(columns = {0: "Horizontal", 1: "Vertical", 2: "Distances"})
    df1[['Horizontal', 'Vertical', 'Distances']] = df1[['Horizontal', 'Vertical', 'Distances']].apply(pd.to_numeric)

    df1 = df1.drop(df1[df1['Distances'] != 1].index)
    df1.reset_index(drop = True, inplace = True)

    return df1

def key_frame_load_data(path, key_slice_index, shape, spacing, block, window):
    try:
        slice_name = os.listdir(path)
        slice_name.sort()

        slices = []
        slice_num = []
        new_slice_name = []

        for i in range(1, len(slice_name)):
            slice_num_i, _ = slice_name[i].split('.')
            slice_num_iminus1, _ = slice_name[i-1].split('.')

            if int(slice_num_i) != int(slice_num_iminus1)+1 or int(slice_num_iminus1) > key_slice_index:
                break
            else:
                new_slice_name.append(slice_name[i-1])
                slice_num.append(slice_num_iminus1)
                slices.append(cv2.imread(path + slice_name[i-1], -1).astype(np.float) - 32768)

        if len(slice_num) < spacing*block or int(slice_num[-1]) != key_slice_index:
            return 'Mistakes Encountered'

        input_slices = []
        for i in range(len(slice_num)-spacing-1, len(slice_num)-spacing*block-2, -spacing):
            new_slice = cv2.resize(slices[i], (shape, shape))
            new_slice = new_slice.reshape([1, shape, shape, 1])
            input_slices.append(data.windowing(new_slice, window))

        input_slices = input_slices[::-1]
        input_slices = np.concatenate(input_slices, axis = -1)

        output_slice = slices[-1]
        output_slice = cv2.resize(output_slice, (shape, shape))
        output_slice = output_slice.reshape([1, shape, shape, 1])
        output_slice = data.windowing(output_slice, window)

        key_slice_name = new_slice_name[-1]

        return input_slices, output_slice, key_slice_name

    except:
        return "Mistakes Encountered"

def rescale(img, window):
    [N, H, W, C] = img.shape
    img = np.reshape(img, [H, W])

    img = (img + 1)/2*(window[1] - window[0]) + window[0]

    img[img > window[1]] = window[1]
    img[img < window[0]] = window[0]

    img = img + 32768
    img = img.astype(np.uint16)

    return img



if __name__ == '__main__':

    #Configurations
    block = 4
    batch = 1
    shape = 512
    chn = 1
    distance = 4
    training_amount = 1200
    testing_amount = 1500
    monitor_patient = 1501

    init_lr = 6e-4
    TRAINFLAG = 0 #1 for training and 0 for testing
    OUTPUT_FOLDER = 'G:/project5/gen_pics/comparison/'
    TESTING_OUTPUT_FOLDER = 'G:/project5/gen_pics/testing_output/'
    CKPT_BASELINE_FOLDER = "G:/project5/ckpt/baseline_ckpt/"
    CKPT_FINAL_FOLDER = "G:/project5/ckpt/"
    BASIC_KEY_FRAME_FOLDER = 'G:/project5/gen_pics/key_frames/basic_pred/'
    FINAL_KEY_FRAME_FOLDER = 'G:/project5/gen_pics/key_frames/new_model/'

    INFO_PATH = 'G:/project5/raw_pics/DL_info_new.csv'
    PATIENT_PATH = 'G:/project5/raw_pics/Images_png/'
    WINDOW = [-1024, 3071]

    # Define placeholders
    Xr_fwd_mn = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, block*chn])
    vessel_masks_Xr = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, block * chn])
    learned_masks_Xr = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, block * chn])
    xn1 = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    vessel_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    learned_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    basic_pred_plac = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    placeholders = [Xr_fwd_mn, xn1, learned_mask, vessel_mask, vessel_masks_Xr, learned_masks_Xr, basic_pred_plac]

    #learning rate decay
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(init_lr, global_step = global_step, decay_steps=10, decay_rate=0.9)
    add_global = global_step.assign_add(1)

    #Get the models from model structure
    basic_pred, gen_var = model.modelStructure.basic_model(Xr_fwd_mn)
    G_loss, var, pred, pred_mask, max_v, min_v = model.modelStructure.structured_model(Xr_fwd_mn, xn1, basic_pred, learned_mask, vessel_mask, vessel_masks_Xr, learned_masks_Xr)
    G_loss_Optimiser = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5, beta2 = 0.999).minimize(G_loss, var_list = var)
    parameters = [G_loss, var, pred, pred_mask, basic_pred, max_v, min_v]

    # Initialise Training Process
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Load the pre-trained baseline model
    saver_pre = tf.train.Saver(gen_var)
    saver_pre.restore(sess, CKPT_BASELINE_FOLDER + 'model.ckpt')

    # Create a saver for other parameters
    trainable_params = var
    saver = tf.train.Saver(trainable_params)

    # Reload other trained parameters
    saver.restore(sess, CKPT_FINAL_FOLDER + 'model.ckpt')

    dl_info = pd.read_csv(INFO_PATH)
    dl_col = dl_info.columns

    # Info Acquisition
    df1 = info_acquisition(INFO_PATH)
    new_df = pd.DataFrame(columns = dl_col)

    for i in range(len(df1)):

        # Acquire Training and Testing Data
        Output = key_frame_load_data(path = PATIENT_PATH + df1['Folder'][i] + '/', key_slice_index = df1['Key_slice_index'][i], shape = shape, spacing = distance, block = block, window = WINDOW)

        if Output == 'Mistakes Encountered':
            continue
        else:
            X, y, key_frame_name = Output
            new_df = new_df.append(dl_info.iloc[i, :])

        Folder_name = df1['Folder'][i]

        basic_pred_val = sess.run(basic_pred, feed_dict = {Xr_fwd_mn: X})

        learned_mask_pred, vessel_mask_pred = data.full_mask_generator(basic_pred_val)
        learned_mask_input, vessel_mask_input = data.full_mask_generator(X, learned_mask_pred)

        body_input = learned_mask_input * X
        vessel_input = (-vessel_mask_input + 1) * X

        feed_dict = {Xr_fwd_mn: X, xn1: y, basic_pred: basic_pred_val,
                     learned_mask: learned_mask_pred, vessel_mask: vessel_mask_pred,
                     vessel_masks_Xr: vessel_mask_input, learned_masks_Xr: learned_mask_input}

        G_loss_val_test = sess.run(G_loss, feed_dict=feed_dict)
        Outimg = sess.run(pred, feed_dict=feed_dict)
        Outpred_mask = sess.run(pred_mask, feed_dict=feed_dict)

        adj_Outpred_mask = rescale(Outpred_mask, WINDOW)
        adj_Outimg = rescale(Outimg, WINDOW)


        # Basic
        basic_name = BASIC_KEY_FRAME_FOLDER + Folder_name + '/'
        create_dir(basic_name)
        state1 = cv2.imwrite(basic_name + key_frame_name, adj_Outpred_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        # New Model
        final_name = FINAL_KEY_FRAME_FOLDER + Folder_name + '/'
        create_dir(final_name)
        state2 = cv2.imwrite(final_name + key_frame_name, adj_Outimg, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

        if state1 == True and state2 == True:
            new_df = new_df.append(dl_info.iloc[i, :])

        print("%i-th frame created and saved, %i frames to finish"%(i, len(df1)-i))

    new_df.to_csv('G:/project5/gen_pics/key_frames/DL_info_updated.csv')
    print("All finished.")