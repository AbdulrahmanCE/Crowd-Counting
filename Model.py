import cv2
import h5py
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal
from keras.applications.vgg16 import VGG16
from skimage.measure import compare_psnr, compare_ssim
from keras.layers import Conv2D, Input, MaxPooling2D

plt.ioff()


class CreateModel:

    def smallize_density_map(self, density_map, stride=1):
        if stride > 1:
            density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int) // stride).tolist(),
                                          dtype=np.float32)
            for r in range(density_map_stride.shape[0]):
                for c in range(density_map_stride.shape[1]):
                    density_map_stride[r, c] = np.sum(
                        density_map[r * stride:(r + 1) * stride, c * stride:(c + 1) * stride])
        else:
            density_map_stride = density_map
        return density_map_stride

    def norm_by_imagenet(self, img):
        if len(img.shape) == 3:
            img = img / 255.0
            img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
            return img
        elif len(img.shape) == 4 or len(img.shape) == 1:
            # In SHA, shape of images varies, so the array.shape is (N, ), that's the '== 1' case.
            imgs = []
            for im in img:
                im = im / 255.0
                im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
                im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
                im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225
                imgs.append(im)
            return np.array(imgs)
        else:
            print('Wrong shape of the input.')
            return None

    def image_preprocessing(self, x, y, flip_hor=False, brightness_shift=False):
        xs, ys = [], []
        for idx_pro in range(x.shape[0]):
            x_, y_ = x[idx_pro], y[idx_pro]
            # preprocessings -----
            if flip_hor:
                x_, y_ = self.flip_horizontally(x_, y_)
            # preprocessings -----
            x_ = self.norm_by_imagenet(x_)
            xs.append(x_)
            ys.append(y_)
        xs, ys = np.array(xs), np.array(ys)
        return xs, ys

    def flip_horizontally(self, x, y):
        to_flip = np.random.randint(0, 2)
        if to_flip:
            x, y = cv2.flip(x, 1), np.expand_dims(cv2.flip(np.squeeze(y), 1), axis=-1)
            # Suppose shape of y is (123, 456, 1), after cv2.flip, shape of y would turn into (123, 456).
        return x, y

    def fix_singular_shape(self, img, unit_len=16):
        """
        Some network like w-net has both N maxpooling layers and concatenate layers,
        so if no fix for their shape as integeral times of 2 ** N, the shape will go into conflict.
        """
        hei_dst, wid_dst = img.shape[0] + (unit_len - img.shape[0] % unit_len), img.shape[1] + (
                unit_len - img.shape[1] % unit_len)
        if len(img.shape) == 3:
            img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
        elif len(img.shape) == 2:
            GT = int(round(np.sum(img)))
            img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
            img = img / (np.sum(img) / GT)
        return img

    def gen_var_from_paths(self, paths, stride=1, unit_len=16):
        vars = []
        format_suffix = paths[0].split('.')[-1]
        if format_suffix == 'h5':
            for ph in paths:
                dm = h5py.File(ph, 'r')['density'].value.astype(np.float32)
                if unit_len:
                    dm = self.fix_singular_shape(dm, unit_len=unit_len)
                dm = self.smallize_density_map(dm, stride=stride)
                vars.append(np.expand_dims(dm, axis=-1))
        elif format_suffix == 'jpg':
            for ph in paths:
                raw = cv2.cvtColor(cv2.imread(ph), cv2.COLOR_BGR2RGB).astype(np.float32)
                if unit_len:
                    raw = self.fix_singular_shape(raw, unit_len=unit_len)
                vars.append(raw)
            # vars = norm_by_imagenet(vars)
        else:
            print('Format suffix is wrong.')
        return np.array(vars)

    def CSRNet(self, input_shape=(None, None, 3)):
        # input shape (None= height, None= width, 3= channel)
        input_flow = Input(shape=input_shape)

        # set the initial random weights of Keras layers (stddev= slandered deviation of the random values to generate)
        dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

        # build cnn layers (CSRNet architecture)
        # front-end (10 conv layers with 3 max pooling)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_flow)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # back-end (6 conv layers without max pooling)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=3, activation='relu',
                   kernel_initializer=dilated_conv_kernel_initializer)(x)

        # output layer
        output_flow = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu',
                             kernel_initializer=dilated_conv_kernel_initializer)(x)

        model = Model(inputs=input_flow, outputs=output_flow)

        # VGG16 model with weights pre-trained on ImageNet
        # imagenet = pre-training on ImageNet
        # include_top = whether to include the 3 fully-connected layers at the top of the network
        front_end = VGG16(weights='imagenet', include_top=False)  # front_end is the vgg model

        # append vgg layers in the weights_front_end list
        weights_front_end = []
        for layer in front_end.layers:
            if 'conv' in layer.name:
                weights_front_end.append(layer.get_weights())

        # set the vgg weights for the first 10 layers in our model
        counter_conv = 0
        for i in range(len(front_end.layers)):
            if counter_conv >= 10:
                break

            if 'conv' in model.layers[i].name:
                model.layers[i].set_weights(weights_front_end[counter_conv])
                model.layers[i].trainable = False
                counter_conv += 1

        return model

    def eval_loss(self, model, x, y, quality=False):
        preds, DM, GT = [], [], []
        losses_SFN, losses_MAE, losses_MAPE, losses_RMSE = [], [], [], []
        for idx_pd in range(x.shape[0]):
            pred = model.predict(np.array([x[idx_pd]]))
            preds.append(np.squeeze(pred))
            DM.append(np.squeeze(np.array([y[idx_pd]])))
            GT.append(round(np.sum(np.array([y[idx_pd]]))))  # To make sure the GT is an integral value
        for idx_pd in range(len(preds)):
            losses_SFN.append(np.mean(np.square(preds[idx_pd] - DM[idx_pd])))  # mean of Frobenius norm
            losses_MAE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]))
            losses_MAPE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]) / GT[idx_pd])
            losses_RMSE.append(np.square(np.sum(preds[idx_pd]) - GT[idx_pd]))

        loss_SFN = np.sum(losses_SFN)
        loss_MAE = np.mean(losses_MAE)
        loss_MAPE = np.mean(losses_MAPE)
        loss_RMSE = np.sqrt(np.mean(losses_RMSE))
        if quality:
            psnr, ssim = [], []
            for idx_pd in range(len(preds)):
                data_range = max([np.max(preds[idx_pd]), np.max(DM[idx_pd])]) - min(
                    [np.min(preds[idx_pd]), np.min(DM[idx_pd])])
                psnr_ = compare_psnr(preds[idx_pd], DM[idx_pd], data_range=data_range)
                ssim_ = compare_ssim(preds[idx_pd], DM[idx_pd], data_range=data_range)
                psnr.append(psnr_)
                ssim.append(ssim_)
            return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE, np.mean(psnr), np.mean(ssim)
        return loss_MAE, loss_RMSE, loss_SFN, loss_MAPE
