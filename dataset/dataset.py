import os
import os.path as osp
import random

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
class TAS500(Dataset):
    """
       TAS500 dataset is employed to load train or val set
       Args:
        root: the TAS500 dataset path,
         tas500v1.1
          ├── train
          ├── train_labels
          ├── train_labels_ids
          ├── val
          ├── val_labels
          ├── val_labels_ids
          ├── test
        crop_size: (512, 1024), only works for 'train' split
        mean: rgb_mean (0.485, 0.456, 0.406)
        std: rgb_mean (0.229, 0.224, 0.225)
        ignore_label: 255
    """
    def __init__(self, root, split='train',crop_size=(512, 1024), mean=(128,128,128),crop=True, scale=True, mirror=True, ignore_label=255):

        self.root = root
        self.mode = split
        self.crop = crop
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.files = []
        # store file paths in dict
        for root_dir, _, files in os.walk(osp.join(root, 'tas500v1.1', split)):
            for file in files:
                img_file = osp.join(root_dir, file)
                label_file = osp.join(root_dir.replace(split,split+'_labels'),file)
                label_ids_file = osp.join(root_dir.replace(split,split+'_labels_ids'),file)
                self.files.append({'img': img_file, 'label': label_file,'ids':label_ids_file, 'name': file})
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # 0-255
        # label = cv2.imread(datafiles["label"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["ids"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        # rescale for multi scale inputs
        if self.scale:
            scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            f_scale = scale[random.randint(0, 5)]
            # f_scale = 0.5 + random.randint(0, 15) / 10.0  # random resize between 0.5 and 2
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB

        if self.crop:
            # random crop (if img size > desired size) or crop after padding(if img size < desired size)
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                            pad_w, cv2.BORDER_CONSTANT,
                                            value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                            pad_w, cv2.BORDER_CONSTANT,
                                            value=(self.ignore_label,))
            else:
                img_pad, label_pad = image, label

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
            image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
            label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        # random flip
        if self.is_mirror: 
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class TASTestDataSet(Dataset):
    """ 
       TASDataSet is employed to load test set
       Args:
        root: the TAS dataset path
    """

    def __init__(self, root, mean=(128, 128, 128),
                 ignore_label=255):
        self.root = root
        self.ignore_label = ignore_label 
        self.mean = mean
        self.files = []
        for root_dir, _, files in os.walk(osp.join(root, 'tas500v1.1', 'test')):
            for file in files:
                img_file = osp.join(root_dir, file)
                self.files.append({'img': img_file, 'name': file})
        print("lenth of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        size = image.shape

        image -= self.mean
        # image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        return image.copy(), np.array(size), name


class TASTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, data_root="", classes=23,
                 inform_data_file="", normVal=1.10):
        """
        Args:
           data_root: directory where the dataset is kept xx/xx/Datasetname
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.data_root = data_root
        self.inform_data_file = inform_data_file

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, root, ignore_label=255):
        """to read the whole train set of current dataset.
        Args:
        root: Dataset root xx/xx/Datasetname
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        # min_val_al = 0
        # max_val_al = 0
        file_paths = []

        # store file paths in dict
        for root_dir, _, files in os.walk(osp.join(root, 'tas500v1.1', "train")):
            for file in files:
                img_file = osp.join(root_dir, file)
                label_file = osp.join(root_dir.replace("train",'train_labels'),file)
                label_ids_file = osp.join(root_dir.replace('train','train_labels_ids'),file)
                file_paths.append({'img': img_file, 'label': label_file,'ids':label_ids_file, 'name': file})

        for file in tqdm(file_paths):
            # we expect the text file to contain the data in following format
            # <RGB Image> <Label Image>
            img_file, label_file, label_ids_file = file["img"], file["label"], file["ids"]

            label_ids_img = cv2.imread(label_ids_file, 0)    
            (unique_values,counts) = np.unique(label_ids_img,return_counts=True)
            ignore_label_mask = unique_values!=ignore_label
            unique_values = unique_values[ignore_label_mask]
            counts = counts[ignore_label_mask]

            max_val = max(unique_values)
            min_val = min(unique_values)
            global_hist[unique_values] += counts
            # max_val_al = max(max_val, max_val_al)
            # min_val_al = min(min_val, min_val_al)

            rgb_img = cv2.imread(img_file)
            self.mean[0] += np.mean(rgb_img[:, :, 0])
            self.mean[1] += np.mean(rgb_img[:, :, 1])
            self.mean[2] += np.mean(rgb_img[:, :, 2])

            self.std[0] += np.std(rgb_img[:, :, 0])
            self.std[1] += np.std(rgb_img[:, :, 1])
            self.std[2] += np.std(rgb_img[:, :, 2])

            if max_val > (self.classes - 1) or min_val < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + label_file)
            no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self,ignore_label=255):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet(root=self.data_root,ignore_label=ignore_label)

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights

            with open(self.inform_data_file, 'wb') as f:
                pickle.dump(data_dict,f)
            return data_dict
        return None