import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import random
import h5py
from scipy.signal import decimate
from scipy.stats import entropy

      


from utils.timefeatures import time_features
import neurokit2 as nk
import heartpy as hp
from tqdm import tqdm
import json


warnings.filterwarnings('ignore')



class G12EC(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path      
        self.seq_len = seq_len      
        self.patch_len = patch_len
        self.flag = flag
        self.dataset_ECG_path = ".../filtered_data_100Hz.npy"
        self.data_ = np.load(self.dataset_ECG_path)
              
              
        self.Y_ = np.load('/.../G12EC/filtered_Y.npy', allow_pickle=True)

        self.label_list = np.loadtxt('.../G12EC/unique_list_filtered.txt',
                                delimiter=',', dtype=str)
        indices = np.where(self.label_list == 'Brady')[0]

        total_samples = self.Y_.shape[0]
        np.random.seed(3223)
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)
              
        train_end = int(0.8 * total_samples)
        val_end = int((0.8 + 0.1) * total_samples)

        if self.flag == "train":
            self.indices = all_indices[:train_end]
            self.data = self.data_[self.indices]
            self.Y = self.Y_[self.indices]
            self.whether_norm = self.Y[:, indices]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")


        elif self.flag == "valid":
            self.indices = all_indices[train_end:val_end]
            self.data = self.data_[self.indices]
            self.Y = self.Y_[self.indices]
            self.whether_norm = self.Y[:, indices]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")

        else:
            self.indices = all_indices[val_end:]
            self.data = self.data_[self.indices]
            self.Y = self.Y_[self.indices]
            self.whether_norm = self.Y[:, indices]

                  
            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")




    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
              
              
              
        return self.data[index, :,:],self.Y[index, :],self.whether_norm[index:index +1,0]
              


class SPH(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.flag = flag
        self.data_o = np.load("/.../data/SPH/Processed_data/X_cut1000.npy")
              
              
              
              
        self.data_ = self.data_o.swapaxes(1, 2)
        self.Y_ = np.load('/.../data/SPH/Processed_data/Y_onehot.npy', allow_pickle=True)
        with open('/.../data/SPH/Processed_data/Y_origin.json','r', encoding='utf-8') as file:
            self.Y_or = json.load(file)
        with open('/.../data/SPH/Processed_data/Y_unique_list.json','r', encoding='utf-8') as file:
            self.label_list = np.ravel(json.load(file))
        with open('/../code_description.json', 'r', encoding='utf-8') as file:
            self.data_dict = json.load(file)
        indices = np.where(self.label_list == '1')[0]

        if self.flag == "train":
            with open('/.../data/SPH/Processed_data/train_index.json', 'r', encoding='utf-8') as file:
                train_index = json.load(file)
            self.data = self.data_[train_index]
            self.Y = self.Y_[train_index]
            self.whether_norm = self.Y[:, indices]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")

        elif self.flag == "valid":
            with open('/.../data/SPH/Processed_data/test_index.json', 'r', encoding='utf-8') as file:
                test_index = json.load(file)
            self.data = self.data_[test_index]
            self.Y = self.Y_[test_index]
            self.whether_norm = self.Y[:, indices]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")

        else:
            with open('/.../data/SPH/Processed_data/test_index.json', 'r', encoding='utf-8') as file:
                test_index = json.load(file)
            self.data = self.data_[test_index]
            self.Y = self.Y_[test_index]
            self.whether_norm = self.Y[:, indices]

                  
            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index, :,:],self.Y[index, :],self.whether_norm[index:index +1]



class MIMIC(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path        
        self.seq_len = seq_len        
        self.patch_len = patch_len
        self.flag = flag
        self.MIMIC_data_path = "/.../data/MIMIC/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/label_json_X_and_Y/"
        self.h5_path = self.MIMIC_data_path + "ecg_filtered.h5"

              
        self.h5f = h5py.File(self.h5_path, "r")

        self.Y_ = np.load(self.MIMIC_data_path + "one_hot_labels.npy", allow_pickle=True)

        with open(
                '/.../data/MIMIC/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/label_json_X_and_Y/label_vocab.json',
                'r', encoding='utf-8') as file:
            label_list = json.load(file)
        self.indices_norm = np.where(np.array(label_list) == 'SINUS RHYTHM')[0]

        total_samples = self.h5f["ecg"].shape[0]
        np.random.seed(3223)
        all_indices = np.arange(total_samples)
        np.random.shuffle(all_indices)

              
        train_end = int(0.8 * total_samples)
        val_end = int((0.8 + 0.1) * total_samples)
        if flag == "train":
            self.indices = all_indices[:train_end]

        elif flag == "val":
            self.indices = all_indices[train_end:val_end]
        else:        
            self.indices = all_indices[val_end:]

              
        has_nan = np.isnan(self.Y_).any()
        print(f"Y数组中是否有NaN: {has_nan}")
        print(f"转换后的数组的维度: {self.Y_.shape}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        true_idx = self.indices[index]

              
        data = self.h5f["ecg"][true_idx]        

              
        data = np.nan_to_num(data, nan=0.0)        

              
        data = data
        Y = self.Y_[true_idx, :]
        whether_norm = self.Y_[true_idx:true_idx + 1, self.indices_norm]
        return data, Y, whether_norm

    def close(self):
              
        self.h5f.close()


class PTBXL(Dataset):
    def __init__(self, root_path, data_path, seq_len, patch_len, flag="train"):
        self.root_path = root_path
        self.data_path = data_path      
        self.seq_len = seq_len      
        self.patch_len = patch_len
        self.flag = flag
        self.dataset_ECG_path = os.path.join(self.root_path,  'X_smooth_100Hz.npy')
        self.data_ = np.load(self.dataset_ECG_path)
              
        self.Y_ = np.load('/.../Y_onehot.npy', allow_pickle=True)

        Y_train = np.load('/.../data/PTBXL/PTBXL_diagnostic/superclass/Y_origin.npy', allow_pickle=True)
        self.label_list = np.loadtxt('/.../Y_unique_list.txt',
                                delimiter=',', dtype=str)
        indices = np.where(self.label_list == 'NORM')[0]
        if self.flag == "train":
            train_indexes = []
            for i in range(9):
                index = np.load(os.path.join(self.root_path, str(i + 1) + "fold_index.npy"))
                train_indexes.append(index)
            train_index = np.concatenate(train_indexes, axis=0)
            self.data = self.data_[train_index]
            self.Y = self.Y_[train_index]
            self.whether_norm =  self.Y[:,indices]

            zero_rows_mask = (np.abs(self.data) < 1e-6).all(axis=1)        
            zero_rows_indices = np.where(zero_rows_mask)[0]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")

        elif self.flag == "valid":
            test_index = np.load(os.path.join(self.root_path, str(9 + 1) + "fold_index.npy"))
            self.data = self.data_[test_index]
            self.Y = self.Y_[test_index]
            self.whether_norm = self.Y[:, indices]
            zero_rows_mask = (np.abs(self.data) < 1e-6).all(axis=1)        
            zero_rows_indices = np.where(zero_rows_mask)[0]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")


        else:
            test_index = np.load(os.path.join(self.root_path, str(9 + 1) + "fold_index.npy"))

            self.data = self.data_[test_index]
            self.Y = self.Y_[test_index]
            self.whether_norm = self.Y[:, indices]

            class_counts = np.sum(self.Y, axis=0)
            for i, count in enumerate(class_counts):
                print(f"类别 {self.label_list[i]} 的样本数量: {count}")



    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        return self.data[index, :,:],self.Y[index, :],self.whether_norm[index:index +1,0]
