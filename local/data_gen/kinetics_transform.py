import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

num_joint = 19
max_frame = 48
num_person_out = 1
num_person_in = 1


class Feeder_kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "RHip"},
    # {9,  "RKnee"},
    # {10, "RAnkle"},
    # {11, "LHip"},
    # {12, "LKnee"},
    # {13, "LAnkle"},
    # {14, "REye"},
    # {15, "LEye"},
    # {16, "REar"},
    # {17, "LEar"},
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=5,
                 num_person_out=2):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        sample_id = [name.split('.')[0] for name in self.sample_name]
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        # has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])

        # ignore the samples which does not has skeleton sequence
        # if self.ignore_empty_sample:
        #     self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
        #     self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_id=index[1]
        index=index[0]
        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        print(sample_path,index)
        # if sample_path == '../data/kinetics_raw/kinetics_val/649.npy':
        #     import pdb;pdb.set_trace()
        with open(sample_path, 'rb') as f:
            video_info = np.load(f)

        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        data_numpy[:3, :, :, 0] = video_info[0,data_id:data_id+3,:,:,0]
        # for ct in range(self.T):
        #     frame_index = ct
        #     data_numpy[0, frame_index, :, 0] = video_info[0,0,frame_index,:,0]
        #     data_numpy[1, frame_index, :, 0] = video_info[0,1,frame_index,:,0]
            
        # for frame_info in video_info['data']:
        #     frame_index = frame_info['frame_index']
        #     for m, skeleton_info in enumerate(frame_info["skeleton"]):
        #         if m >= self.num_person_in:
        #             break
        #         pose = skeleton_info['pose']
        #         score = skeleton_info['score']
        #         data_numpy[0, frame_index, :, m] = pose[0::2]
        #         data_numpy[1, frame_index, :, m] = pose[1::2]
        #         data_numpy[2, frame_index, :, m] = score

        # centralization
        # data_numpy[0:2] = data_numpy[0:2] - 0.5
        # data_numpy[1:2] = -data_numpy[1:2]
        # data_numpy[0][data_numpy[2] == 0] = 0
        # data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        label = self.label[index] 
        # assert (self.label[index] == label)

        # sort by score
        # sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        # for t, s in enumerate(sort_index):
        #     data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                    #    0))
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame, data_id=0):
    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)
    # import pdb;pdb.set_trace()
    for i, s in enumerate(tqdm(sample_name)):
        data, label = feeder[(i,data_id)]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/kinetics_raw')
    parser.add_argument(
        '--out_folder', default='../data/kinetics')
    arg = parser.parse_args()

    part = ['val', 'train' , 'test']
    data_name=['joint','velocity','acceleration','boneL','boneA']
    data_index = [0,3,6,9,12]
    for dn in range(len(data_name)):
        for p in part:
            print('kinetics ', p)
            if not os.path.exists(arg.out_folder):
                os.makedirs(arg.out_folder)
            data_path = '{}/kinetics_{}'.format(arg.data_path, p)
            label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
            data_out_path = '{}/{}_data_{}.npy'.format(arg.out_folder, p,data_name[dn])
            label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)

            gendata(data_path, label_path, data_out_path, label_out_path, data_id=data_index[dn])
