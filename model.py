import numpy as np
import tensorflow as tf
import tensorlayer as tl

import tensorflow.keras as keras

class modelComponents():

    @staticmethod
    def tensorial_neural_block(x, new_shape1, new_shape2, new_shape3, scope, actv = tf.nn.sigmoid, isIN = True, isActv = True):


        with tf.variable_scope(scope):

            [N, H, W, C] = x.get_shape().as_list()
            var1 = tf.Variable(initial_value = tf.random_normal(shape=[N, H, C, new_shape3], mean=0, stddev=1), name = 'var1')
            x = tf.matmul(x, var1)
            x = tf.transpose(x, [0, 1, 3, 2])

            [N, H, W, C] = x.get_shape().as_list()
            var2 = tf.Variable(initial_value = tf.random_normal(shape=[N, H, C, new_shape2], mean=0, stddev=1), name = 'var2')
            x = tf.matmul(x, var2)
            x = tf.transpose(x, [0, 3, 2, 1])

            [N, H, W, C] = x.get_shape().as_list()
            var3 = tf.Variable(initial_value = tf.random_normal(shape=[N, H, C, new_shape1], mean=0, stddev=1), name = 'var3')
            x = tf.matmul(x, var3)
            x = tf.transpose(x, [0, 3, 1, 2])

            # x = tf.matmul(x, var1)
            # x = tf.tensordot(x, var2, axes = 2)
            # x = tf.tensordot(x, var3, axes = 3)

            if isIN:
                x = tf.contrib.layers.instance_norm(x)

            if isActv:
                x = actv(x)

            return x

    @staticmethod
    def conv2d_block(x, filters, scope, pad = 1, kernels = (3, 3), strides = (2, 2), actv = tf.nn.relu, isIN = True, isActv = True, padding = 'VALID'):

        x = tf.pad(x, paddings = ([0, 0], [pad, pad], [pad, pad], [0, 0]), mode = 'REFLECT')
        conv1 = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernels, strides = strides, padding = padding,
                                 use_bias = True, name = scope)
        if isIN == True:
            conv1 = tf.contrib.layers.instance_norm(conv1)

        if isActv == True:
            conv1 = actv(conv1)
        return conv1

    @staticmethod
    def pooling2d_block(x, pool_size = (2, 2), strides = (2, 2)):
        pool1 = tf.layers.MaxPooling2D(pool_size = pool_size, strides = strides)(x)
        return pool1

    @staticmethod
    def convLSTM2D(x, filter, scope, seq = False, pad = 1, actv = tf.nn.leaky_relu, kernels = (3, 3), strides = (2, 2), isIN = True, isActv = True):

        [N, H, W, C] = x.get_shape().as_list()

        x = tf.transpose(x, (0, 3, 1, 2))
        x = x[:, :, :, :, tf.newaxis]

        x = tf.pad(x, [[0, 0], [0, 0], [pad, pad], [pad, pad], [0, 0]], mode = 'REFLECT')
        input_shape = x.get_shape().as_list()
        conv1 = keras.layers.ConvLSTM2D(filters = filter, kernel_size = kernels, strides = strides, data_format = 'channels_last'
                                        , input_shape = input_shape, padding = 'VALID', return_sequences = seq, name = scope)(x)
        if isIN is True:
            conv1 = tf.contrib.layers.instance_norm(conv1)

        if isActv is True:
            conv1 = actv(conv1)

        if seq is True:
            conv1 = tf.transpose(conv1, (0, 2, 3, 1, 4))
            conv1 = tf.reshape(conv1, shape = [N, H, W, C])

        return conv1

    @staticmethod
    def upsampling2d_block(x, filter, scope, kernels=(3, 3), strides=(2, 2), padding='SAME', actv=tf.nn.relu):

        up1 = tf.layers.conv2d_transpose(inputs = x, filters = filter, kernel_size = kernels, strides = strides, padding = padding,
                                         use_bias = True, name = scope+'_conv_transpose')
        up1 = tf.contrib.layers.instance_norm(up1)
        up1 = actv(up1)
        return up1

    @staticmethod
    def res2d_block(x, scope, filters):
        padded = tf.pad(x, paddings = ([0, 0], [1, 1], [1, 1], [0, 0]), mode = 'REFLECT')
        conv1 = tf.layers.conv2d(inputs = padded, filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'VALID',
                                 name = scope+'conv1')
        conv1 = tf.contrib.layers.instance_norm(conv1)
        conv1 = tf.nn.relu(conv1)

        conv1 = tf.pad(conv1, paddings = ([0, 0], [1, 1], [1, 1], [0, 0]), mode = 'REFLECT')
        conv2 = tf.layers.conv2d(inputs = conv1, filters = filters, kernel_size = (3, 3), strides = (1, 1), padding = 'VALID',
                                 name = scope+'conv2')
        conv2 = tf.contrib.layers.instance_norm(conv2)

        return x + conv2

    @staticmethod
    def dense_block(x, layer_filters, final_filters, dense_layers, scope, actv = tf.nn.relu, final_actv = tf.nn.relu, reuse = False):
        comp = modelComponents()

        pre_input = x
        # pre_input = tf.nn.relu(pre_input)
        # pre_input = tf.contrib.layers.instance_norm(pre_input)

        with tf.variable_scope(scope, reuse = reuse):
            for i in range(dense_layers):

                dense = comp.conv2d_block(x=pre_input, filters=layer_filters, strides=(1, 1), actv = actv, scope = 'dense%i'%i)
                pre_input = tf.concat([pre_input, dense], axis=-1)

            output = comp.conv2d_block(x=pre_input, filters=final_filters, strides=(1, 1), actv = final_actv, scope='output')

        return output

    @staticmethod
    def gaussian_kernel(size=3, sigma=1.5):
        x_points = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
        y_points = x_points[::-1]
        xs, ys = np.meshgrid(x_points, y_points)
        kernel = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        return kernel / kernel.sum()

    @staticmethod
    def laplacian_kernel():
        return tf.constant([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

    @staticmethod
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):

        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return tf.convert_to_tensor(h, dtype = tf.float32)

    @staticmethod
    def filter2(x, kernel, mode='SAME'):
        return tf.nn.conv2d(x, tf.image.rot90(kernel, 2), strides = [1, 1, 1, 1], padding=mode)

class modelStructure():

    @staticmethod
    def dualAttModule(x, param_h, param_v, scope, reuse = False):

        param_softmax_v = param_v
        param_softmax_h = param_h

        [N, H, W, C] = x.get_shape().as_list()

        with tf.variable_scope('%s_vertical_att'%scope, reuse = reuse):
            att_v = tf.layers.conv2d(x, kernel_size = (1, 1), filters = C, name = 'conv_att_v')
            att_v_reshaped = tf.reshape(att_v, shape = [N, H, W, C], name = 'conv_att_v_reshaped')

            att_v_reshaped = att_v_reshaped/param_softmax_v
            att_v_softmax = tf.nn.softmax(logits = att_v_reshaped, axis = 1, name = 'conv_att_v_softmax')

            att_v_softmax_max = tf.reduce_max(att_v_softmax, axis = 1, name = 'conv_att_v_softmax_max')

            att_v_softmax_norm = att_v_softmax/att_v_softmax_max

            x = x * att_v_softmax_norm
            x_reshaped = tf.reshape(tensor=x, shape=[N, H, W, C])

        with tf.variable_scope('%s_horizontal_att'%scope, reuse = reuse):
            att_h = tf.layers.conv2d(x_reshaped, kernel_size = (1, 1), filters = C, name = 'conv_att_h')

            att_h_flatten = tf.reshape(att_h, shape = [N, H*W, C], name = 'conv_att_h_flatten')
            att_h_flatten = att_h_flatten/param_softmax_h

            att_h_flatten_softmax = tf.nn.softmax(logits = att_h_flatten, axis = -1, name = 'conv_att_h_flatten_softmax')
            att_h_flatten_softmax_max = tf.reduce_max(att_h_flatten_softmax, axis = -1, name = 'conv_att_h_flatten_softmax_max')
            att_h_flatten_softmax_max = tf.reshape(att_h_flatten_softmax_max, shape = [N, H*W, 1])

            att_h_softmax_norm = att_h_flatten_softmax/att_h_flatten_softmax_max
            att_h_softmax_norm_reshaped = tf.reshape(tensor=att_h_softmax_norm, shape=[N, H, W, C])

            att_h = att_h * att_h_softmax_norm_reshaped

            att_reshaped = tf.reshape(tensor=att_h, shape=[N, H, W, C])

        return att_reshaped

    @staticmethod
    def dense_net(x, scope, final_filter = 1, reuse = False):

        comp = modelComponents()
        structure = modelStructure()

        with tf.variable_scope(scope, reuse = reuse):
            dense1 = comp.dense_block(x, layer_filters = 10, final_filters = 32, dense_layers = 20, scope = 'dense_net1', reuse = reuse)
            dense1 = comp.conv2d_block(dense1, filters = 16, pad = 1, kernels = (3, 3), strides = (1, 1), scope = 'dense_net1_compress', actv = tf.nn.leaky_relu)

            dense2 = comp.dense_block(dense1, layer_filters=10, final_filters = 64, dense_layers = 20, scope = 'dense_net2', reuse = reuse)
            dense2 = comp.conv2d_block(dense2, filters=32, pad=1, kernels=(3, 3), strides=(1, 1), scope='dense_net2_compress', actv = tf.nn.leaky_relu)

            dense3 = comp.dense_block(dense2, layer_filters=10, final_filters = 128, dense_layers = 20, scope = 'dense_net3', reuse = reuse)
            dense3 = comp.conv2d_block(dense3, filters=64, pad=1, kernels=(3, 3), strides=(1, 1), scope='dense_net3_compress', actv = tf.nn.leaky_relu)

            dense6 = comp.dense_block(dense3, layer_filters=5, final_filters = 1024, dense_layers = 20, scope = 'dense_net6', reuse = reuse)
            dense6 = comp.conv2d_block(dense6, filters=final_filter, pad=1, kernels=(3, 3), strides=(1, 1), scope='dense_net6_compress', isIN = False, isActv = False)
            # dense6 = comp.pooling2d_block(dense6) #h/32, w/32

        return tf.nn.tanh(dense6)

    @staticmethod
    def unet_generator(x, scope, final_actv = None, reuse = False):

        comp = modelComponents()

        with tf.variable_scope(scope, reuse = reuse):
            conv1_1 = comp.conv2d_block(x, filters=64, pad = 3, kernels = (7, 7), strides = (1, 1), scope = 'conv1_1') # 256
            conv2_1 = comp.conv2d_block(conv1_1, filters = 64, pad = 1, kernels = (3, 3), scope = 'conv2_1') # 128
            conv3_1 = comp.conv2d_block(conv2_1, filters = 256, pad = 1, kernels = (3, 3), scope = 'conv3_1') # 64
            conv4_1 = comp.conv2d_block(conv3_1, filters = 512, pad = 1, kernels = (3, 3), scope = 'conv4_1')  # 32

            res = conv4_1
            for i in range(1, 10):
                res = comp.res2d_block(res, scope = 'res_%i'%i, filters = 512)

            # Conv_transpose
            res = tf.concat([res, conv4_1], axis = -1)
            deconv1_1 = comp.upsampling2d_block(res, filter=256, scope='deconv1_1', actv = tf.nn.leaky_relu) # 64
            deconv1_1 = tf.concat([deconv1_1, conv3_1], axis = -1)
            deconv2_1 = comp.upsampling2d_block(deconv1_1, filter=128, scope='deconv2_1', actv = tf.nn.leaky_relu) # 128
            deconv2_1 = tf.concat([deconv2_1, conv2_1], axis = -1)
            deconv3_1 = comp.upsampling2d_block(deconv2_1, filter=64, scope='deconv3_1', actv = tf.nn.leaky_relu) # 256
            deconv3_1 = tf.concat([deconv3_1, conv1_1], axis = -1)

            # large_output = comp.tensorial_neural_block(x = deconv2_1, new_shape1 = 256, new_shape2 = 256, new_shape3 = 1, scope = 'large_output', isIN = False, isActv = False)
            large_output = comp.conv2d_block(deconv3_1, filters = 1, pad = 3, kernels = (7, 7), strides = (1, 1), scope = 'large_output', isIN = False, isActv = False)

            if final_actv is not None:
                large_output = final_actv(large_output)

        return large_output

    @staticmethod
    def generator(x, scope, final_layer = True, reuse = False):

        comp = modelComponents()

        with tf.variable_scope(scope, reuse = reuse):
            conv1_1 = comp.conv2d_block(x, filters=64, pad = 3, kernels = (7, 7), strides = (1, 1), scope = 'conv1_1')
            conv2_1 = comp.conv2d_block(conv1_1, filters = 64, pad = 1, kernels = (3, 3), scope = 'conv2_1')
            conv3_1 = comp.conv2d_block(conv2_1, filters = 256, pad = 1, kernels = (3, 3), scope = 'conv3_1')

            res = conv3_1
            for i in range(1, 10):
                res = comp.res2d_block(res, scope = 'res_%i'%i, filters = 256)

            # Conv_transpose
            deconv1_1 = comp.upsampling2d_block(res, filter=128, scope='deconv1_1', actv = tf.nn.leaky_relu)
            deconv2_1 = comp.upsampling2d_block(deconv1_1, filter=64, scope='deconv2_1', actv = tf.nn.leaky_relu)
            large_output = comp.conv2d_block(deconv2_1, filters=1, pad=3, kernels=(7, 7), strides=(1, 1), scope='large_output', isIN=False, isActv=False)


            if final_layer:
                large_output_after = tf.nn.tanh(large_output)

        return large_output_after

    @staticmethod
    def discriminator(x, scope, reuse = False):

        comp = modelComponents()
        with tf.variable_scope(scope, reuse=reuse):
            dis1 = comp.conv2d_block(x, filters=8, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis1', isIN=False, actv=tf.nn.leaky_relu)
            dis2 = comp.conv2d_block(dis1, filters=32, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis2', actv=tf.nn.leaky_relu)
            dis3 = comp.conv2d_block(dis2, filters=64, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis3', actv=tf.nn.leaky_relu)
            dis4 = comp.conv2d_block(dis3, filters=1, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis4', isIN=False, isActv=False)

            output = tf.nn.sigmoid(dis4)

            return output

    @staticmethod
    def feature_matching_discriminator(x, scope, reuse = False):

        comp = modelComponents()
        with tf.variable_scope(scope, reuse=reuse):
            dis1 = comp.conv2d_block(x, filters=8, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis1', isIN=False, actv=tf.nn.leaky_relu)
            dis2 = comp.conv2d_block(dis1, filters=32, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis2', actv=tf.nn.leaky_relu)
            dis3 = comp.conv2d_block(dis2, filters=64, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis3', actv=tf.nn.leaky_relu)
            dis4 = comp.conv2d_block(dis3, filters=1, pad=1, kernels=(3, 3), strides=(2, 2), scope='dis4', isIN=False, isActv=False)

            output = tf.nn.sigmoid(dis4)

            return output, [dis1, dis2, dis3, dis4]

    @staticmethod
    def GAN_Loss(D_real, D_fake):
        D_loss = tf.reduce_mean(tf.square(D_real - 1.)) + tf.reduce_mean(tf.square(D_fake))
        G_loss = tf.reduce_mean(tf.square(D_fake - 1.))
        return D_loss, G_loss

    @staticmethod
    def abs_loss(true, pred, mask = None):
        if mask is not None:
            return tf.reduce_sum(mask*true-mask*pred)/tf.reduce_sum(mask)
        else:
            return tf.losses.absolute_difference(true, pred)

    @staticmethod
    # Laplician of Gaussian Losses that exerts the importances of Boundary
    def LoG_Loss(gt, gen, mask = None, delta = 0.01, abs = False):
        comp = modelComponents()
        structure = modelStructure()

        # Make Gaussian Kernel with desired specs.
        gauss_kernel = comp.gaussian_kernel()
        laplace_kernel = comp.laplacian_kernel()

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv3d` signature.
        gauss_kernel_new = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        laplace_kernel_new = laplace_kernel[:, :, tf.newaxis, tf.newaxis]

        # Convolve.
        gt = tf.nn.conv2d(gt, gauss_kernel_new, strides=[1, 1, 1, 1], padding="SAME")
        gt = tf.nn.conv2d(gt, laplace_kernel_new, strides=[1, 1, 1, 1], padding="SAME")

        gen = tf.nn.conv2d(gen, gauss_kernel_new, strides=[1, 1, 1, 1], padding="SAME")
        gen = tf.nn.conv2d(gen, laplace_kernel_new, strides=[1, 1, 1, 1], padding="SAME")

        # Abs loss
        if abs is True:
            if mask is not None:
                return structure.abs_loss(gt, gen, mask)
            else:
                return structure.abs_loss(gt, gen)

    @staticmethod
    def feature_matching_loss(fea_real, fea_fake, delta = 0.01):
        structure = modelStructure()
        loss = 0
        for i in range(len(fea_fake)):
            loss += tf.reduce_mean(structure.abs_loss(fea_fake[i], fea_real[i]))
        return loss

    @staticmethod
    def huber(true, pred, delta=0.01):
        loss = tf.where(tf.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2), delta * tf.abs(true - pred) - 0.5 * (delta ** 2))
        return tf.reduce_sum(loss)

    @staticmethod
    def metric_func(im1, im2):

        comp = modelComponents()

        k1 = 0.01
        k2 = 0.03
        win_size = 11
        L = 1

        [N, H, W, C] = im1.get_shape().as_list()
        # im1 = tf.reshape(im1, [H, W])
        # im2 = tf.reshape(im2, [H, W])
        #
        # M, N = im1.shape
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = comp.matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / tf.reduce_sum(tf.reduce_sum(window))

        window = window[:, :, tf.newaxis, tf.newaxis]

        if im1.dtype == tf.uint8:
            im1 = tf.double(im1)
        if im2.dtype == tf.uint8:
            im2 = tf.double(im2)

        mu1 = comp.filter2(im1, window, 'VALID')
        mu2 = comp.filter2(im2, window, 'VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = comp.filter2(im1 * im1, window, 'VALID') - mu1_sq
        sigma2_sq = comp.filter2(im2 * im2, window, 'VALID') - mu2_sq
        sigmal2 = comp.filter2(im1 * im2, window, 'VALID') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return tf.reduce_mean(tf.reduce_mean(ssim_map))


    @staticmethod
    def basic_model(X):
        comp = modelComponents()
        structure = modelStructure()

        y_basic = structure.generator(x = X, scope = 'gen', reuse = False)
        basic_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'gen')

        return y_basic, basic_var

    @staticmethod
    def structured_model(X, y, body_inputs, true_body, vessel_inputs, true_vessel, y_basic, body_mask, vessel_mask, old_window, body_window, vessel_window):
        comp = modelComponents()
        structure = modelStructure()

        [N, H, W, C] = X.get_shape().as_list()

        with tf.variable_scope('model'):

            # Prepare Training Materials
            pred_vessel_basic = y_basic * (-vessel_mask + 1) # -1~1
            pred_body_basic = y_basic * body_mask #-1~1
            background = y_basic - pred_body_basic - pred_vessel_basic #-1~1

            # Refined Vessel & Refined Body
            refined_vessel = structure.unet_generator(vessel_inputs, scope = 'further_refinement_vessel', final_actv = tf.nn.tanh, reuse = False) # -1~1
            # refined_body = structure.dense_net(vessel_inputs, final_filter = 1, scope = 'further_refinement_vessel', reuse = False)  # -1~1
            refined_body = structure.dense_net(body_inputs, final_filter = 1, scope = 'further_refinement_body', reuse = False) # -1~1

            # Adjust them to the original window [-1024, 3071] for comparison
            refined_body_adj = (refined_body+1)/2*(body_window[1] - body_window[0]) + body_window[0]
            refined_body_adj = (refined_body_adj - old_window[0])/(old_window[1] - old_window[0]) *2 - 1

            refined_vessel_adj = (refined_vessel+1)/2*(vessel_window[1] - vessel_window[0]) + vessel_window[0]
            refined_vessel_adj = (refined_vessel_adj - old_window[0])/(old_window[1] - old_window[0]) *2 - 1

            # Final pred after fusion
            final_pred = refined_body_adj * body_mask + refined_vessel_adj*(-vessel_mask + 1) + background  # -1~1

            # Loss Function
            Loss_vessel_abs = tf.losses.absolute_difference(true_vessel*(1-vessel_mask), refined_vessel*(1-vessel_mask)) # -1~1 and -1~1
            Loss_body_abs = tf.losses.absolute_difference(true_body*body_mask, refined_body*body_mask) # -1~1 and -1~1

            # Loss_vessel_huber = structure.huber(true_vessel*(1-vessel_mask), refined_vessel*(1-vessel_mask))
            # Loss_body_huber = structure.huber(true_body*body_mask, refined_body*body_mask)

            Loss_img_ssim =  - structure.metric_func(y, final_pred) # -1~1 and -1~1

            Losses = Loss_vessel_abs + Loss_body_abs + Loss_img_ssim
            # Losses = Loss_vessel_abs + Loss_body_abs

        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'model')

        return Losses, var, final_pred, y_basic, refined_body*body_mask, true_body*body_mask