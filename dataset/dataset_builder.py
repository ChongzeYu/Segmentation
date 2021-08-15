import os
import pickle
from torch.utils import data
from .dataset import TAS500,TASTrainInform

def build_dataset_train(dataset, input_size, batch_size, random_scale, random_mirror, num_workers):
    data_root = '/home/chongze/SegmentationAuto/'
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "TAS500":
            dataCollect = TASTrainInform(data_root, 23, inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports only one dataset: TAS5001.1, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    trainLoader = data.DataLoader(
            TAS500(data_root, split='train', crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

    valLoader = data.DataLoader(
            TAS500(data_root, split='val', crop_size=input_size, mean=datas['mean'],scale=False, mirror=False),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

    return datas, trainLoader, valLoader



def build_dataset_test(dataset, num_workers, none_gt=False):
    data_dir = os.path.join('./dataset/', dataset)
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = TAS500(data_dir, 19,
                                                inform_data_file=inform_data_file)
        elif dataset == 'camvid':
            dataCollect = TAS500(data_dir, 11, 
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if none_gt:
            testLoader = data.DataLoader(
                TAS500(data_dir, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
        testLoader = data.DataLoader(
            TAS500(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    return datas, testLoader