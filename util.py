import tensorflow as tf
import numpy as np
import pandas as pd
import time

import matplotlib
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import convolve2d

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import model
# from data import *
from new_data import *

class Util_set():

    @staticmethod
    def info_acquisition(INFO_PATH):
        # Acquire information
        dl_info = pd.read_csv(INFO_PATH)

        dl_info.drop_duplicates(subset=['Patient_index', 'Study_index', 'Series_ID'], keep='first', inplace=True)
        df1 = dl_info['File_name'].str.split('_', expand=True)
        df1['Window'] = dl_info['DICOM_windows']
        df1['Folder'] = df1[0].str.cat([df1[1], df1[2]], sep='_')
        df1['Spacing_mm_px_'] = dl_info['Spacing_mm_px_']
        df1 = df1.drop(columns=[0, 1, 2, 3])

        df2 = df1['Spacing_mm_px_'].str.split(', ', expand=True)
        df1 = pd.merge(df1, df2, left_index=True, right_index=True)

        df1 = df1.drop(columns=['Spacing_mm_px_'])
        df1.reset_index(drop=True, inplace=True)
        df1 = df1.rename(columns={0: "Horizontal", 1: "Vertical", 2: "Distances"})
        df1[['Horizontal', 'Vertical', 'Distances']] = df1[['Horizontal', 'Vertical', 'Distances']].apply(pd.to_numeric)

        df1 = df1.drop(df1[df1['Distances'] != 1].index)
        df1.reset_index(drop=True, inplace=True)

        return df1

    @staticmethod
    def rescalePic(x):#rescale picture to size of [H, W] and range of [0, 255]
        [N, H, W, C] = x.shape
        x = np.reshape(x, newshape=[H, W])
        return 255*(x+1)/2

    @staticmethod
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    @staticmethod
    def filter2(x, kernel, mode='same'):
        return convolve2d(x, np.rot90(kernel, 2), mode=mode)

    @staticmethod
    def metric_func(type, im1, im2):

        util_component = Util_set()

        if type == 'PSNR':
            MSE = np.mean(np.square(im1 - im2))
            MAXI = np.max(im1)
            return 10*np.log10(MAXI**2/MSE)

        if type == 'SSIM':
            k1 = 0.01
            k2 = 0.03
            win_size = 11
            L = 255

            if not im1.shape == im2.shape:
                raise ValueError("Input Imagees must have the same dimensions")
            if len(im1.shape) > 2:
                raise ValueError("Please input the images with 1 channel")

            M, N = im1.shape
            C1 = (k1 * L) ** 2
            C2 = (k2 * L) ** 2
            window = util_component.matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
            window = window / np.sum(np.sum(window))

            if im1.dtype == np.uint8:
                im1 = np.double(im1)
            if im2.dtype == np.uint8:
                im2 = np.double(im2)

            mu1 = util_component.filter2(im1, window, 'valid')
            mu2 = util_component.filter2(im2, window, 'valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = util_component.filter2(im1 * im1, window, 'valid') - mu1_sq
            sigma2_sq = util_component.filter2(im2 * im2, window, 'valid') - mu2_sq
            sigmal2 = util_component.filter2(im1 * im2, window, 'valid') - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return np.mean(np.mean(ssim_map))

    @staticmethod
    def testing_module(OUTPUT_FOLDER, TRAINFLAG, data, num_i, parameters, placeholders, sess):

        WINDOW = [-1024, 3071]

        util_component = Util_set()

        Loss, var, y_final, y_basic, test1, test2 = parameters
        X, y, body_inputs, true_body, vessel_inputs, true_vessel, vessel_mask, body_mask, y_basic_plac = placeholders
        X_test, y_test = data

        [N, H, W, C] = X_test.shape

        y_basic_val = sess.run(y_basic, feed_dict = {X: X_test})
        body_mask_pred, vessel_mask_pred = maskGenerator(y_basic_val)

        # plt.imshow(y_basic_val.reshape([256, 256]), cmap = 'gray')
        # plt.show()

        body_mask_pred = body_mask_pred.reshape([N, H, W, 1])
        vessel_mask_pred = vessel_mask_pred.reshape([N, H, H, 1])

        vessel_mask_X = []
        body_mask_X = []

        for i in range(5):
            vessel_mask_X.append(vessel_mask_pred)
            body_mask_X.append(body_mask_pred)

        vessel_mask_X = np.concatenate(vessel_mask_X, axis = -1)
        body_mask_X = np.concatenate(body_mask_X, axis = -1)

        full_inputs = np.concatenate([X_test, y_basic_val], axis = -1)
        vessel_inputs_val = full_inputs * (1 - vessel_mask_X)
        body_inputs_val = full_inputs * body_mask_X

        true_vessel_val = y_test * (1 - vessel_mask_pred)
        true_body_val = y_test * body_mask_pred

        vessel_inputs_val = (vessel_inputs_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
        body_inputs_val = (body_inputs_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
        true_vessel_val = (true_vessel_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]
        true_body_val = (true_body_val + 1) / 2 * (WINDOW[1] - WINDOW[0]) + WINDOW[0]

        max_vessel = -200
        min_vessel = -1000

        max_body = 720
        min_body = -360

        vessel_inputs_val = windowing(vessel_inputs_val, [min_vessel, max_vessel], mode = 'Postprocessing')
        body_inputs_val = windowing(body_inputs_val, [min_body, max_body], mode = 'Postprocessing')

        true_vessel_val = windowing(true_vessel_val, [min_vessel, max_vessel], mode = 'Postprocessing')
        true_body_val = windowing(true_body_val, [min_body, max_body], mode = 'Postprocessing')

        vessel_inputs_val = vessel_inputs_val * (1 - vessel_mask_X)
        body_inputs_val = body_inputs_val * body_mask_X
        true_vessel_val = true_vessel_val * (1 - vessel_mask_pred)
        true_body_val = true_body_val * body_mask_pred


        feed_dict_input = {X: X_test, y: y_test, y_basic_plac: y_basic_val,
                           body_inputs: body_inputs_val, vessel_inputs: vessel_inputs_val,
                           true_vessel: true_vessel_val, true_body: true_body_val,
                           body_mask: body_mask_pred, vessel_mask: vessel_mask_pred}

        feed_dict_output = {X: X_test, y_basic_plac: y_basic_val,
                            body_inputs: body_inputs_val, vessel_inputs: vessel_inputs_val,
                            true_vessel: true_vessel_val, true_body: true_body_val,
                            body_mask: body_mask_pred, vessel_mask: vessel_mask_pred}

        G_loss_val_test = sess.run(Loss, feed_dict=feed_dict_input)
        Outimg = sess.run(y_final, feed_dict=feed_dict_output)
        Outtest1 = sess.run(test1, feed_dict = feed_dict_output)
        Outtest2 = sess.run(test2, feed_dict = feed_dict_output)

        [N, H, W, C] = Outimg.shape

        # Post processing
        adj_Outimg = util_component.rescalePic(Outimg)
        adj_Realimg = util_component.rescalePic(y_test)
        adj_Outbasic = util_component.rescalePic(y_basic_val)

        # Metrics
        PSNR = util_component.metric_func(type='PSNR', im1=adj_Realimg, im2=adj_Outimg)
        PSNR_interim = util_component.metric_func(type = 'PSNR', im1 = adj_Realimg, im2 = adj_Outbasic)
        SSIM = util_component.metric_func(type='SSIM', im1=adj_Realimg, im2=adj_Outimg)
        SSIM_interim = util_component.metric_func(type = 'SSIM', im1 = adj_Realimg, im2 = adj_Outbasic)
        MAE = np.asscalar(np.mean(np.abs(adj_Outimg - adj_Realimg)))
        MAE_interim = np.asscalar(np.mean(np.abs(adj_Outbasic - adj_Realimg)))

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 5, 1)
        ax1.imshow(adj_Outimg, cmap='gray')
        ax1.set_title('Refined')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(adj_Outbasic, cmap='gray')
        ax2.set_title('Baseline')
        ax2.set_axis_off()

        ax3 = fig.add_subplot(1, 5, 3)
        ax3.imshow(adj_Realimg, cmap='gray')
        ax3.set_title('Ground Truth')
        ax3.set_axis_off()

        ax4 = fig.add_subplot(1, 5, 4)
        ax4.imshow(Outtest1.reshape([H, W]), cmap='gray')
        ax4.set_title('Test')
        ax4.set_axis_off()

        ax5 = fig.add_subplot(1, 5, 5)
        ax5.imshow(Outtest2.reshape([H, W]), cmap='gray')
        ax5.set_title('Test2')
        ax5.set_axis_off()

        # garbage collection
        del adj_Realimg
        del adj_Outimg
        del adj_Outbasic
        del Outtest1
        del Outtest2

        plt.savefig(OUTPUT_FOLDER + '%i_iteration_%.4f_PSNR_%.4f_SSIM_%.4f_MAE.png' % (num_i, PSNR, SSIM, MAE))

        plt.close(fig)
        del fig
        del ax1
        del ax2
        del ax3
        del ax4
        del ax5


        if TRAINFLAG == 1:
            print("The testing loss at %i epoch is: %.4f, PSNR = %.4f, PSNR of basic pred = %.4f, SSIM = %.4f, SSIM of basic pred = %.4f, MAE = %.4f, MAE of basic pred = %.4f" %
                      (num_i, G_loss_val_test, PSNR, PSNR_interim, SSIM, SSIM_interim, MAE, MAE_interim))

        elif TRAINFLAG == 0:
            return (Outimg+1)/2, (y_test+1)/2, PSNR, PSNR_interim, SSIM, SSIM_interim, MAE, MAE_interim