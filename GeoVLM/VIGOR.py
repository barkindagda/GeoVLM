import os
import numpy as np
import random
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x

        return img_shift[:,:,:fov_index]

def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class VIGOR(Dataset):
    def __init__(self, mode='', root='PATH TO VIGOR DATASET', captions_root='PATH TO VIGOR CAPTIONS', same_area=True, print_bool=False, polar='', args=None, image_encoder=None):
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root
        self.captions_root = captions_root
        self.polar = polar
        self.mode = mode
        self.image_encoder = image_encoder

        self.transform_query = input_transform(size=(384, 768))
        self.transform_reference = input_transform(size=(384, 384))

        self.same_area = same_area
        label_root = 'splits'

        if same_area:
            self.train_city_list = ['SanFrancisco','Chicago']
            self.test_city_list = ['SanFrancisco','Chicago']
        else:
            self.train_city_list = ['Seattle','NewYork']
            self.test_city_list = ['Chicago','SanFrancisco']

        # Load train satellite list
        self.train_sat_list = []
        self.train_sat_index_dict = {}
        idx = 0
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.strip()))
                    self.train_sat_index_dict[line.strip()] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        if print_bool:
            print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        # Load test satellite list
        self.test_sat_list = []
        self.test_sat_index_dict = {}
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.strip()))
                    self.test_sat_index_dict[line.strip()] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        if print_bool:
            print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        # Load train panorama list
        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt' if self.same_area else 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]] if data[i] in self.train_sat_index_dict else -1)
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if label[0] != -1:
                        if label[0] not in self.train_sat_cover_dict:
                            self.train_sat_cover_dict[label[0]] = [idx]
                        else:
                            self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', train_label_fname, idx)
        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        if print_bool:
            print('Train grd loaded, data_size: {}'.format(self.train_data_size))

        # Load test panorama list
        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        idx = 0
        for city in self.test_city_list:
            test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt' if self.same_area else 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]] if data[i] in self.test_sat_index_dict else -1)
                    label = np.array(label).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.test_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.test_label.append(label)
                    self.test_delta.append(delta)
                    if label[0] != -1:
                        if label[0] not in self.test_sat_cover_dict:
                            self.test_sat_cover_dict[label[0]] = [idx]
                        else:
                            self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', test_label_fname, idx)
        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        if print_bool:
            print('Test grd loaded, data size: {}'.format(self.test_data_size))

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())

        # Load captions
        self.train_captions_re = {}
        self.train_captions_query = {}
        self.test_captions_re = {}
        self.test_captions_query = {}

        for city in self.train_city_list:
            captions_re_path = os.path.join(self.captions_root, city, 'captions_reference.csv')
            captions_query_path = os.path.join(self.captions_root, city, 'captions_query.csv')

            if os.path.exists(captions_re_path):
                df_re = pd.read_csv(captions_re_path)
                self.train_captions_re.update(dict(zip(df_re.image_name, df_re.caption)))

            if os.path.exists(captions_query_path):
                df_query = pd.read_csv(captions_query_path)
                self.train_captions_query.update(dict(zip(df_query.image_name, df_query.caption)))

        for city in self.test_city_list:
            captions_re_path = os.path.join(self.captions_root, city, 'captions_reference.csv')
            captions_query_path = os.path.join(self.captions_root, city, 'captions_query.csv')

            if os.path.exists(captions_re_path):
                df_re = pd.read_csv(captions_re_path)
                self.test_captions_re.update(dict(zip(df_re.image_name, df_re.caption)))

            if os.path.exists(captions_query_path):
                df_query = pd.read_csv(captions_query_path)
                self.test_captions_query.update(dict(zip(df_query.image_name, df_query.caption)))

    def __getitem__(self, index):
        if self.mode == 'train':
            query_img = Image.open(self.train_list[index])
            query_img = self.transform_query(query_img)

            query_image_name = os.path.basename(self.train_list[index])
            query_caption = self.train_captions_query.get(query_image_name, "")

            return query_img, query_caption, torch.tensor(self.train_label[index][0])

        elif self.mode == 'test_query':
            img_query = Image.open(self.test_list[index])
            img_query = self.transform_query(img_query)

            query_image_name = os.path.basename(self.test_list[index])
            query_caption = self.test_captions_query.get(query_image_name, "")

            return img_query, query_caption, torch.tensor(self.test_label[index][0])

        elif self.mode == 'test_reference':
            img_reference = Image.open(self.test_sat_list[index]).convert('RGB')
            img_reference = self.transform_reference(img_reference)

            reference_image_name = os.path.basename(self.test_sat_list[index])
            reference_caption = self.test_captions_re.get(reference_image_name, "")

            return img_reference, reference_caption, torch.tensor(index)

        else:
            print('Not implemented mode:', self.mode)
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_list)
        elif self.mode == 'test_query':
            return len(self.test_list)
        elif self.mode == 'test_reference':
            return len(self.test_sat_list)
        else:
            print('Not implemented mode:', self.mode)
            raise Exception

    def get_all_reference_images(self):
        all_ref_images = []
        for img_path in self.train_sat_list:
            img = Image.open(img_path).convert('RGB')
            img = self.transform_reference(img)
            all_ref_images.append(img)
        return torch.stack(all_ref_images)

    def get_reference_captions(self, indices):
        captions = []
        for idx in indices.flatten():
            img_name = os.path.basename(self.train_sat_list[idx])
            caption = self.train_captions_re.get(img_name, "")
            captions.append(caption)
        return captions
