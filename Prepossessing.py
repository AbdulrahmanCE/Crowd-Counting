import os
import cv2
import glob
import h5py
import scipy
import shutil
import tensorflow as tf
from scipy import integrate
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.io import loadmat, savemat


class Prepossessing:
    def generatepaths(self, img_path):
        img_paths = []
        gt_paths = []

        for path in glob.glob(os.path.join(img_path, '*.jpg')):
            img_paths.append(path)
            gt_paths.append(path.replace('.jpg', '.mat').replace('images', 'ground-truth'))

        return img_paths, gt_paths

    def createdir(self, dirs):
        for folder in dirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def patches(self, img_paths, gt_paths, newimages_paths, newgroundtruth_paths):
        count = 1

        print("Crop images to smalll patches:")
        for img_path, gt_path in tqdm(zip(img_paths, gt_paths)):

            img = cv2.imread(img_path)
            pts = loadmat(gt_path)

            new_y = int(img.shape[0] / 2)
            new_x = int(img.shape[1] / 2)

            # Start crop patches
            for y in range(0, img.shape[0] - int(img.shape[0] / 2), int(img.shape[0] / 6)):
                for x in range(0, img.shape[1] - int(img.shape[1] / 2), int(img.shape[1] / 6)):

                    # Create new mat files
                    new_mat = {'__header__': str(datetime.now()), '__version__': "Python 3.6",
                               '__globals__': ['annPoints'],
                               'annPoints': list([])}

                    for mat in pts["annPoints"]:
                        if x < mat[0] < (x + new_x) and y < mat[1] < (y + new_y):
                            new_mat['annPoints'].append(list([mat[0] - x, mat[1] - y]))

                    x = int(x)
                    y = int(y)
                    crop_img = img[y:(y + new_y), x:(x + new_x)]

                    # Save new images and mat files for train data

                    cv2.imwrite(os.path.join(newimages_paths, 'crop_Image{}.jpg'.format(count)), crop_img)
                    savemat(os.path.join(newgroundtruth_paths, 'crop_Image{}.mat'.format(count)), new_mat)

                    count += 1

    def gen_density_map_gaussian(self, im, points, sigma=4):
        """
        func: generate the density map
        """
        density_map = np.zeros(im.shape[:2], dtype=np.float32)
        h, w = density_map.shape[:2]
        num_gt = np.squeeze(points).shape[0]
        if num_gt == 0:
            return density_map
        if sigma == 4:
            # Adaptive sigma in CSRNet.
            leafsize = 2048
            tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
            distances, _ = tree.query(points, k=4)
        for idx_p, p in enumerate(points):
            p = np.round(p).astype(int)
            p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
            gaussian_radius = sigma * 2 - 1
            if sigma == 4:
                # Adaptive sigma in CSRNet.
                sigma = max(int(np.sum(distances[idx_p][1:4]) * 0.1), 1)
                gaussian_radius = sigma * 3
            gaussian_map = np.multiply(
                cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma),
                cv2.getGaussianKernel(int(gaussian_radius * 2 + 1), sigma).T
            )
            x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
            # cut the gaussian kernel
            if p[1] < gaussian_radius:
                x_left = gaussian_radius - p[1]
            if p[0] < gaussian_radius:
                y_up = gaussian_radius - p[0]
            if p[1] + gaussian_radius >= w:
                x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
            if p[0] + gaussian_radius >= h:
                y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
            gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
            if np.sum(gaussian_map):
                gaussian_map = gaussian_map / np.sum(gaussian_map)
            density_map[
            max(0, p[0] - gaussian_radius):min(h, p[0] + gaussian_radius + 1),
            max(0, p[1] - gaussian_radius):min(w, p[1] + gaussian_radius + 1)
            ] += gaussian_map
        density_map = density_map / (np.sum(density_map / num_gt))
        return density_map

    def calculatedesnity(self, new_img_paths, new_gt_paths):
        count = 1
        print("Calculate density maps:")
        for img_path, gt_path in tqdm(zip(new_img_paths, new_gt_paths)):

            # Load .mat files
            pts = loadmat(gt_path)
            img = cv2.imread(img_path)

            sigma = 3
            k = np.zeros((img.shape[0], img.shape[1]))
            gt = pts["annPoints"]

            for i in range(len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            DM = self.gen_density_map_gaussian(k, gt, sigma=sigma)
            dm_path = img_path.replace('.jpg', '.h5').replace('images', 'density-maps')
            with h5py.File(dm_path, 'w') as hf:
                hf['density'] = DM

            count += 1

    def separate_data(self):

        images = 'Data/haramData/images'
        images_paths, groundtruth_paths = self.generatepaths(images)

        newdata = 'Data/NewData'
        newimages = 'Data/NewData/images'
        newgroundtruth = 'Data/NewData/ground-truth'
        newdensitymap = 'Data/NewData/density-maps'

        train = 'Data/NewData/train'
        train_img = 'Data/NewData/train/images'
        train_gt = 'Data/NewData/train/ground-truth'
        train_dm = 'Data/NewData/train/density-maps'

        test = 'Data/NewData/test'
        test_img = 'Data/NewData/test/images'
        test_gt = 'Data/NewData/test/ground-truth'
        test_dm = 'Data/NewData/test/density-maps'

        folders = [newdata, newimages, newgroundtruth, newdensitymap, train, train_img, train_gt, train_dm, test,
                   test_img, test_gt, test_dm]

        self.createdir(folders)
        # self.patches(images_paths, groundtruth_paths, newimages, newgroundtruth)
        new_images_paths, new_groundtruth_paths = self.generatepaths(newimages)
        # self.calculatedesnity(new_images_paths, new_groundtruth_paths)

        maxval = 0
        minval = 100000000
        total = 0

        for gt_path in new_groundtruth_paths:
            mat = loadmat(gt_path)["annPoints"]
            length = len(mat)
            if length > maxval:
                maxval = length
            if length < minval:
                minval = length
            total += length

        avrg = total / len(gt_path)

        print('max value: ', maxval)
        print('min value: ', minval)
        print('total value: ', total)
        print('average: ', avrg)

        # shutil.rmtree(newimages)
        # shutil.rmtree(newgroundtruth)
        # shutil.rmtree(newdensitymap)

        counter = 0
        img_parts = []
        for path in glob.glob(os.path.join(newimages, '*.jpg')):
            img_parts.append(path)

        for path in img_parts:
            if counter < len(img_parts) * 0.80:
                shutil.move(path, train_img)
                shutil.move(path.replace('.jpg', '.mat').replace('images', 'ground-truth'), train_gt)
                shutil.move(path.replace('.jpg', '.h5').replace('images', 'density-maps'), train_dm)
            else:
                shutil.move(path, test_img)
                shutil.move(path.replace('.jpg', '.mat').replace('images', 'ground-truth'), test_gt)
                shutil.move(path.replace('.jpg', '.h5').replace('images', 'density-maps'), test_dm)
            counter += 1
        # shutil.rmtree(part_A_img)
        # shutil.rmtree(part_A_gt)
        # shutil.rmtree(part_A_dm)
        #
        # shutil.rmtree(part_B_img)
        # shutil.rmtree(part_B_gt)
        # shutil.rmtree(part_B_dm)
