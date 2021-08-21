import os
import pickle
from torch.utils import data
from .dataset import TAS500,TASTrainInform,TASTestDataSet

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
            TAS500(data_root, split='val',crop=False,mean=datas['mean'],scale=False, mirror=False),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

    return datas, trainLoader, valLoader



def build_dataset_test(dataset, num_workers, mode="test"):
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

    if mode=="test":
        testLoader = data.DataLoader(
            TASTestDataSet(data_root, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif mode=="val":
        testLoader = data.DataLoader(
            TAS500(data_root, split='val', mean=datas['mean'],scale=False, mirror=False, crop=False),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    elif mode=="train":
        testLoader = data.DataLoader(
            TAS500(data_root, split='train', mean=datas['mean'],scale=False, mirror=False, crop=False),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError("wrong mode of build_dataset_test")

    return datas, testLoader