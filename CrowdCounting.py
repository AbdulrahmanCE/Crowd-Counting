import datetime
import os
import cv2
import sys
import glob
import shutil
import tqdm
import numpy as np
import pandas as pd
from time import time, ctime
import matplotlib.pyplot as plt
from keras.engine.saving import model_from_json
from keras.optimizers import Adam, SGD
from Model import CreateModel
from Prepossessing import Prepossessing
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

plt.ioff()


def main():
    # data = Prepossessing()
    # data.separate_data()

    train = CreateModel()
    # Settings
    net = 'CSRNet'
    dataset = 'mixed_data'

    weights_dir = 'weights_' + dataset + "_{}".format('EXP_1')

    # weights_dir = 'weights_' + dataset
    if not os.path.exists(weights_dir):
        print('create dir: ', weights_dir)
        os.makedirs(weights_dir)

    # Generate paths of (train, test) x (img, dm)
    train_img_paths = []
    train_dm_paths = []
    test_img_paths = []
    test_dm_paths = []

    train_paths = 'Data/NewData/train/images'
    test_paths = 'Data/NewData/test/images'

    # train_B_paths = 'Data/NewData/part_A/train/images'
    # test_B_paths = 'Data/NewData/part_A/test/images'

    for path in glob.glob(os.path.join(train_paths, '*.jpg')):
        train_img_paths.append(path)
        train_dm_paths.append(path.replace('.jpg', '.h5').replace('images', 'density-maps'))
        break

    for path in glob.glob(os.path.join(test_paths, '*.jpg')):
        test_img_paths.append(path)
        test_dm_paths.append(path.replace('.jpg', '.h5').replace('images', 'density-maps'))
        break

    train_x = train.gen_var_from_paths(train_img_paths[:], unit_len=None)
    train_y = train.gen_var_from_paths(train_dm_paths[:], stride=8, unit_len=None)
    print('Train data size:', train_x.shape[0], train_y.shape[0], len(train_img_paths))

    test_x = train.gen_var_from_paths(test_img_paths[:], unit_len=None)
    test_y = train.gen_var_from_paths(test_dm_paths[:], stride=8, unit_len=None)
    test_x = train.norm_by_imagenet(test_x)
    print('Test data size:', test_x.shape[0], test_y.shape[0], len(test_img_paths))

    # if os.path.exists(weights_dir):
    #     shutil.rmtree(weights_dir)
    # os.makedirs(weights_dir)

    # Settings of network
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # run in the first GPU
    LOSS = 'MSE'
    optimizer = Adam(lr=1e-5)

    # Create our model
    model = train.CSRNet(input_shape=(None, None, 3))
    model.compile(optimizer=optimizer, loss='MSE')
    model.summary()  # print the model details

    # create the model directory
    if not os.path.exists('models'):
        os.makedirs('models')

    # save the model in json format with vgg wights
    with open('./models/{}.json'.format(net), 'w') as fout:
        fout.write(model.to_json())

    # Settings of training
    batch_size = 1
    epoch = 250
    val_rate = 0.5
    val_rate_dec = {'mixed_data': [80, 70], 'B': [9, 8.5]}
    len_train = train_x.shape[0]
    num_iter = int((len_train - 0.1) // batch_size + 1)
    best_values = {'mae': 1e5, 'rmse': 1e5, 'sfn': 1e5, 'mape': 1e5}
    losses = [[1e5, 1e5, 1e5, 1e5]]

    # Settings of display
    dis_idx = 16 if dataset == 'mixed_data' else 0
    dis_path = test_img_paths[dis_idx]
    dis_x = test_x[dis_idx]
    dis_y = test_y[dis_idx]
    dis_lim = (5, 35) if dataset == 'mixed_data' else (40, 150)
    time_st = time()

    # Training iterations
    for ep in tqdm.tqdm(range(epoch)):
        for idx_train in range(0, len_train, batch_size):

            # set the progress details of the training
            dis_epoch = str(ep + 1) + '-' + str(idx_train + 1) + '_' + str(len_train)

            # load each raw from x , y train variables (raw = image)
            x, y = train_x[idx_train:idx_train + batch_size], train_y[idx_train:idx_train + batch_size]

            # Preprocessing on raw images (flip and normalize each images)
            x, y = train.image_preprocessing(
                x, y,
                flip_hor=True
            )

            # start training the model for each image (one by one)
            # print(idx_train)
            model.fit(x, y, batch_size=1, verbose=1)

            idx_val = (idx_train / batch_size + 1)

            # Eval losses and save models
            if idx_val % (num_iter * val_rate) == 0:

                # predict the ground-truth for test images
                loss = train.eval_loss(model, test_x, test_y, quality=False)

                # choose the model with the best number of errors
                if loss[0] < val_rate_dec[dataset][0]:
                    val_rate = min(val_rate, 0.25)
                if loss[0] < val_rate_dec[dataset][1]:
                    val_rate = min(val_rate, 0.1)
                losses.append(loss)

                # save weights of the model with the best number of errors
                if (loss[0] < best_values['mae']) or (loss[0] == best_values['mae'] and loss[1] < best_values['rmse']):
                    model.save_weights(os.path.join(weights_dir, '{}_best.hdf5'.format(net)))

                # check the current loss if it is less tha the best losses then save the result
                for idx_best in range(len(loss)):
                    if loss[idx_best] < best_values[list(best_values.keys())[idx_best]]:
                        best_values[list(best_values.keys())[idx_best]] = loss[idx_best]
                        to_save = True
                if to_save:
                    path_save = os.path.join(weights_dir, ''.join([
                        net,
                        '_MAE', str(round(loss[0], 3)), '_RMSE', str(round(loss[1], 3)),
                        '_SFN', str(round(loss[2], 3)), '_MAPE', str(round(loss[3], 3)),
                        '_epoch', str(ep + 1), '-', str(idx_val), '.hdf5'
                    ]))
                    model.save_weights(path_save)
                    to_save = False

            # display the progress of training (epochs, errors, and time consuming)
            time_consuming = time() - time_st
            print('In epoch {}, with MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r'.format(
                dis_epoch, np.round(np.array(losses)[-1, :], 2),
                int(time_consuming / 60), int(time_consuming - int(time_consuming / 60) * 60)
            ))
            sys.stdout.flush()

    # Save records
    losses = np.array(losses[1:])
    pd.DataFrame(losses).to_csv('{}/loss.csv'.format(weights_dir), index=False, header=['MAE', 'RMSE', 'SFN', 'MAPE'])
    losses_MAE, losses_RMSE, losses_SFN, losses_MAPE = losses[:, 0], losses[:, 1], losses[:, 2], losses[:, 3]
    plt.plot(losses_MAE, 'r')
    plt.plot(losses_RMSE, 'b')
    multiplier = int(round(dis_lim[0] / (np.min(losses_SFN) + 0.1)))
    plt.plot(losses_SFN * multiplier, 'g')
    plt.legend(['MAE', 'RMSE', 'SFN*{}'.format(multiplier)])
    plt.ylim(dis_lim)
    plt.title('Val_losses in {} epochs'.format(epoch))
    plt.savefig('{}/{}_val_loss.png'.format(weights_dir, net))
    plt.show()

    # Rename weights_dir by the trainging end time, to prevent the careless deletion or overwriting
    end_time_of_train = '-'.join(ctime().split()[:-2])
    suffix_new_dir = '_{}_{}_bestMAE{}_{}'.format(dataset, LOSS, str(round(best_values['mae'], 3)), end_time_of_train)
    weights_dir_neo = 'weights' + suffix_new_dir
    shutil.move(weights_dir, weights_dir_neo)

    # Analysis on results
    # dis_idx = 16 if dataset == 'B' else 0
    # weights_dir_neo = 'weights_B_MSE_BCE_bestMAE7.846_Sat-May-18'
    # model = model_from_json(open('models/{}.json'.format(net), 'r').read())
    model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
    ct_preds = []
    ct_gts = []
    for i in range(len(test_x[:])):
        if i % 100 == 0:
            print('{}/{}'.format(i, len(test_x)))
        i += 0
        test_x_display = np.squeeze(test_x[i])
        test_y_display = np.squeeze(test_y[i])
        path_test_display = test_img_paths[i]
        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        ct_pred = np.sum(pred)
        ct_gt = round(np.sum(test_y_display))
        ct_preds.append(ct_pred)
        ct_gts.append(ct_gt)

    plt.plot(ct_preds, 'r>')
    plt.plot(ct_gts, 'b+')
    plt.legend(['ct_preds', 'ct_gts'])
    plt.title('Pred vs GT')
    plt.show()
    error = np.array(ct_preds) - np.array(ct_gts)
    plt.plot(error)
    plt.title('Pred - GT, mean = {}, MAE={}'.format(
        str(round(np.mean(error), 3)),
        str(round(np.mean(np.abs(error)), 3))
    ))
    plt.show()
    idx_max_error = np.argsort(np.abs(error))[::-1]

    # Show the 5 worst samples
    for worst_idx in idx_max_error[:5].tolist() + [dis_idx]:
        test_x_display = np.squeeze(test_x[worst_idx])
        test_y_display = np.squeeze(test_y[worst_idx])
        path_test_display = test_img_paths[worst_idx]
        pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
        fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
        ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
        ax_x_ori.set_title('Original Image')
        ax_y.imshow(test_y_display, cmap=plt.cm.jet)
        ax_y.set_title('Ground_truth: ' + str(np.sum(test_y_display)))
        ax_pred.imshow(pred, cmap=plt.cm.jet)
        ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
        plt.show()

    # Generate losses and the image quality
    # model = model_from_json(open('models/{}.json'.format(net), 'r').read())
    # model.load_weights('{}/{}_best.hdf5'.format('weights_', net))
    # from utils_callback import eval_loss
    lossMAE, lossRMSE, lossSFN, lossMAPE, PSNR, SSIM = train.eval_loss(
        model, test_x, test_y, quality=True
    )
    print(lossMAE, lossRMSE, lossSFN, lossMAPE, PSNR, SSIM)


if __name__ == "__main__":
    main()
