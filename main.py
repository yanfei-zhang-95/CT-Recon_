import tensorflow as tf
import pandas as pd
import json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import model
from new_data import *
from util import *

import cv2


if __name__ == '__main__':
    #Configurations
    block = 4
    batch = 1
    shape = 256
    chn = 1
    new_spacing = 4 # Distance between 2 slices, 4 for 4mm

    init_lr = 3e-4
    TRAINFLAG = 1 #1 for training and 0 for testing
    OUTPUT_FOLDER = 'F:/project5/gen_pics/comparison/'
    TESTING_OUTPUT_FOLDER = 'F:/project5/gen_pics/testing_output/'
    CKPT_BASELINE_FOLDER = "F:/project5/ckpt/baseline_ckpt/"
    CKPT_FINAL_FOLDER = "F:/project5/ckpt/"

    INFO_PATH = 'F:/project5/raw_pics/DL_info.csv'
    PATIENT_PATH = 'F:/project5/raw_pics/Images_png/'
    WINDOW = [-1024, 3071]
    BODY_WINDOW = [-360, 720]
    VESSEL_WINDOW = [-1000, -200]

    window = WINDOW

    #Auxiliary Utilities
    util_component = Util_set()

    # Define placeholders
    X = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, block*chn])
    y = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    body_inputs = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, (block+1)*chn])
    true_body = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    vessel_inputs = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, (block+1)*chn])
    true_vessel = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    vessel_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    body_mask = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])
    y_basic_plac = tf.placeholder(dtype = tf.float32, shape = [batch, shape, shape, chn])

    placeholders = [X, y, body_inputs, true_body, vessel_inputs, true_vessel, vessel_mask, body_mask, y_basic_plac]

    #learning rate decay
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(init_lr, global_step = global_step, decay_steps=10, decay_rate=0.9)
    add_global = global_step.assign_add(1)

    #Get the models from model structure
    y_basic_pre, basic_var = model.modelStructure.basic_model(X)
    Loss, var, y_final, y_basic, test1, test2 = model.modelStructure.structured_model(X = X, y = y,
                                                                              body_inputs = body_inputs, true_body = true_body,
                                                                              vessel_inputs = vessel_inputs, true_vessel = true_vessel,
                                                                              y_basic = y_basic_pre,
                                                                              body_mask = body_mask, vessel_mask = vessel_mask,
                                                                              old_window = WINDOW, body_window = BODY_WINDOW, vessel_window = VESSEL_WINDOW)
    Loss_Optimiser = tf.train.AdamOptimizer(learning_rate = lr, beta1 = 0.5, beta2 = 0.999).minimize(Loss, var_list = var)
    parameters = [Loss, var, y_final, y_basic, test1, test2]

    # Testing Data
    # df_testing = info_acquisition(INFO_PATH, 'Testing')
    # testing_list = list_generator(df_testing, block = block, new_spacing = new_spacing)
    #
    # random_testing_list = testing_list.copy()
    # random.shuffle(random_testing_list) # Shuffle the training procedure to make a better and universal learning
    #
    # file_name = 'random_testing_list.json'
    # json_read_write(file_name = file_name, target = random_testing_list, mode = 'save')

    random_testing_list = json_read_write(file_name = 'random_testing_list.json', mode = 'read')

    # Training Data
    # df_training = info_acquisition(INFO_PATH, 'Training')
    # training_list = list_generator(df_training, block = block, new_spacing = new_spacing)
    #
    # random_training_list = training_list.copy()
    # random.shuffle(random_training_list) # Shuffle the training procedure to make a better and universal learning
    #
    # file_name = 'random_training_list.json'
    # json_read_write(file_name = file_name, target = random_training_list, mode = 'save')

    random_training_list = json_read_write(file_name = 'random_training_list.json', mode = 'read')


    # Initialise Training Process
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Load the pre-trained baseline model
    saver_pre = tf.train.Saver(basic_var)
    saver_pre.restore(sess, CKPT_BASELINE_FOLDER + 'model.ckpt')

    # Create a saver for other parameters
    trainable_params = var
    saver = tf.train.Saver(trainable_params)

    sess.run(lr)
    sess.run(add_global)
    global_step_val = 0


    #Training Process
    if TRAINFLAG == 1:
        for i in range(len(random_training_list)):

            data = slice_generator(random_training_list[i], shape = shape, PATIENT_PATH = PATIENT_PATH, WINDOW = WINDOW)

            if data is not None:
                X_train, y_train = data
            else:
                print('Acquired something that does not exist.')
                continue

            y_basic_val = sess.run(y_basic, feed_dict = {X: X_train})
            body_mask_pred, vessel_mask_pred = maskGenerator(y_basic_val)

            body_mask_pred = body_mask_pred.reshape([batch, shape, shape, chn])
            vessel_mask_pred = vessel_mask_pred.reshape([batch, shape, shape, chn])

            # Prepare Training Materials
            vessel_mask_X = []
            body_mask_X = []

            for j in range(block+1):
                vessel_mask_X.append(vessel_mask_pred)
                body_mask_X.append(body_mask_pred)

            vessel_mask_X = np.concatenate(vessel_mask_X, axis = -1)
            body_mask_X = np.concatenate(body_mask_X, axis = -1)

            full_inputs = np.concatenate([X_train, y_basic_val], axis = -1)
            vessel_inputs_val = full_inputs*(1-vessel_mask_X)
            body_inputs_val = full_inputs*body_mask_X

            true_vessel_val = y_train * (1 - vessel_mask_pred)
            true_body_val = y_train * body_mask_pred


            vessel_inputs_val = (vessel_inputs_val + 1)/2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
            body_inputs_val = (body_inputs_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]

            true_vessel_val = (true_vessel_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
            true_body_val = (true_body_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]

            max_vessel = -200
            min_vessel = -1000

            max_body = 720
            min_body = -360

            vessel_inputs_val = windowing(vessel_inputs_val, VESSEL_WINDOW, mode = 'Postprocessing')
            body_inputs_val = windowing(body_inputs_val, BODY_WINDOW, mode = 'Postprocessing')
            true_vessel_val = windowing(true_vessel_val, VESSEL_WINDOW, mode = 'Postprocessing')
            true_body_val = windowing(true_body_val, BODY_WINDOW, mode = 'Postprocessing')

            vessel_inputs_val = vessel_inputs_val*(1-vessel_mask_X)
            body_inputs_val = body_inputs_val*body_mask_X
            true_vessel_val = true_vessel_val*(1-vessel_mask_pred)
            true_body_val = true_body_val*body_mask_pred

            if np.sum(np.abs(true_vessel_val)) > 1000:

                _, Loss_val = sess.run([Loss_Optimiser, Loss], feed_dict = {X: X_train, y: y_train,
                                                                            vessel_inputs: vessel_inputs_val, body_inputs: body_inputs_val,
                                                                            true_body: true_body_val, true_vessel: true_vessel_val,
                                                                            y_basic_plac: y_basic_val,
                                                                            body_mask: body_mask_pred, vessel_mask: vessel_mask_pred})
            else:
                print('Too few details, skip this iteration: %i iter'%(i))
                # continue

            #Run a single testing procedure and update learning rate
            if i % 100 == 0:
                testing_len = len(random_testing_list)
                randnum = int(np.random.uniform(1, testing_len, 1))

                data = slice_generator(random_testing_list[randnum], shape = shape, PATIENT_PATH = PATIENT_PATH, WINDOW = WINDOW)

                if data is None:
                    print('Acquired something that does not exist.')
                    continue

                util_component.testing_module(OUTPUT_FOLDER, TRAINFLAG, data, i, parameters, placeholders, sess)

            if i%1000 == 0:
              _, global_step_val = sess.run([add_global, global_step])

            if i%5000 == 0:
                rate = sess.run(lr)
                print("Decay rate adjusted at %d global step and the new decay rate is %.8f"%(global_step_val, rate))


        #Save training parameters at the end of training
        saver.save(sess, CKPT_FINAL_FOLDER+"model.ckpt")
        print('Training finished, model saved to '+CKPT_FINAL_FOLDER)

    #Testing process
    elif TRAINFLAG == 0:
        #Reload other trained parameters
        saver.restore(sess, CKPT_FINAL_FOLDER+'model.ckpt')

        #Collect PSNR, SSIM, and MAE
        PSNR = []
        SSIM = []
        MAE = []

        PSNR_basic = []
        SSIM_basic = []
        MAE_basic = []

        #Runing testing procedure for each folder
        for i in range(len(random_testing_list)):
            data = slice_generator(random_testing_list[i], shape = shape, PATIENT_PATH = PATIENT_PATH, WINDOW = WINDOW)

            if data is None:
                print('Acquired something that does not exist.')
                continue

            util_component.testing_module(OUTPUT_FOLDER, TRAINFLAG, data, i, parameters, placeholders, sess)