import pickle
import glob
import numpy as np
import cv2
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


label_name = ['airplane', 'automobile', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse',
              'ship', 'truck']

train_list = glob.glob(
    '/Users/aihemaitiabulipizi/Documents/ahmatjan/cv-aihe/CV-personal/dataset/cifar10/data_batch_*'
)
test_list = glob.glob(
    '/Users/aihemaitiabulipizi/Documents/ahmatjan/cv-aihe/CV-personal/dataset/cifar10/test_batch'
)
# print(train_list)
# print(test_list)
save_path = '/Users/aihemaitiabulipizi/Documents/ahmatjan/cv-aihe/CV-personal/dataset/cifar10/TRAIN'
test_save_path = '/Users/aihemaitiabulipizi/Documents/ahmatjan/cv-aihe/CV-personal/dataset/cifar10/TEST'
# train data img
for l in train_list:
    l_dict = unpickle(l)
    print('train:', l_dict.keys())

    for im_index, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_index]
        im_name = l_dict[b'filenames'][im_index]

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))
        #
        # cv2.imshow('im_data',cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(save_path, im_label_name)):
            os.mkdir("{}/{}".format(save_path, im_label_name))

        train_img = "{}/{}/{}".format(save_path,
                                      im_label_name,
                                      im_name.decode("utf-8"))
        if not os.path.exists(train_img):
            cv2.imwrite(train_img, im_data)

# test data img
for l in test_list:
    l_dict = unpickle(l)
    print('test:', l_dict.keys())

    for im_index, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_index]
        im_name = l_dict[b'filenames'][im_index]

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))
        #
        # cv2.imshow('im_data',cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists("{}/{}".format(test_save_path, im_label_name)):
            os.mkdir("{}/{}".format(test_save_path, im_label_name))

        test_img = "{}/{}/{}".format(test_save_path,
                                     im_label_name,
                                     im_name.decode("utf-8"))
        if not os.path.exists(test_img):
            cv2.imwrite(test_img, im_data)

