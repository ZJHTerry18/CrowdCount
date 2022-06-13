# -*- coding: utf-8 -*-

import torch.utils.data as data
from . import common
import os
import json
from PIL import Image
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree


class JingtouTrainDataset(data.Dataset):
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.data_path = data_path
        self.datasetname = datasetname
        # self.imshape = (1080,1920)

        self.file_id = []
        self.info = []
        self.gtdict = {}
        self.dot_list = []
        self.den_list = []

        self.gtmode = argv['gt_mode']
        self.list_file = argv['list_file']

        with open(self.list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                self.info.append(splited[1:3]) # lum, crowd level

        self.read_annot(os.path.join(data_path, 'annotations', 'train.json')) # self.gtdict: {'000001': {'name':'05-zhaji-1', 'size':(1080,1920),'ann':[[0.0,1.0],[1.0,2.0]...]}}
        self.file_id = self.gtdict.keys()

        self.imgfileTemp = os.path.join(data_path, 'images', 'train', '{}')
        # self.matfileTemp = os.path.join(data_path, 'mats', '{}.mat')
        # self.gtfileTemp = os.path.join(data_path, self.gtmode, '{}.png')
        if self.gtmode == 'dot':
            self.gen_dot()
        elif self.gtmode == 'den':
            self.gen_den(sigma_default=3, adapt=False)

        self.num_samples = len(self.file_id)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.dot_transform = None
        if 'dot_transform' in argv.keys():
            self.dot_transform = argv['dot_transform']
        self.den_transform = None
        if 'den_transform' in argv.keys():
            self.den_transform = argv['den_transform']
        
        if self.mode == 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode == 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')   

    
    def __getitem__(self, index):
        img = self.read_image(index)
        gt = self.read_gt(index)
        
        if self.gt_mode == 'dot':
            gt = self.dot_list[index]
        elif self.gt_mode == 'den':
            gt = self.den_list[index]
      
        if self.main_transform is not None:
            img, gt = self.main_transform(img, gt) 
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gtmode == 'dot':
            if self.dot_transform is not None:
                gt = self.dot_transform(gt)
        elif self.gtmode == 'den':
            if self.den_transform is not None:
                gt = self.den_transform(gt)

        if self.mode == 'train':    
            return img, gt 
        elif self.mode == 'val':
            attributes_pt = torch.from_numpy(np.array(
                list(map(int, self.info[index]))
            ))
            return img, gt, attributes_pt
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples


    def read_image(self,index):
        img_path = self.imgfileTemp.format(self.gtdict[self.file_id[index]])
        img = Image.open(img_path)

        return img

    def read_gt(self,index):
        pass
    
    def read_annot(self, json_path):
        with open(self.list_file, 'r') as f:
            data_names = f.readlines()
        data_names = [x.split()[0] for x in data_names]

        with open(json_path, 'r') as f:
            dat = json.load(f)

        for iminfo in dat['images']:
            if iminfo['file_name'] in data_names:
                self.gtdict[iminfo['id']] = {
                    'name': iminfo['file_name'],
                    'size': (iminfo['height'], iminfo['width']),
                    'annPoints': []
                }
        
        for annot in dat['annotations']:
            if annot['image_id'] in self.gtdict.keys():
                self.gtdict[annot['image_id']]['annPoints'].append(annot['headPoints'])


    def read_cr_annot(self, json_path):
        with open(json_path, 'r') as f:
            dat = json.load(f)
        
        for annot in dat['annotations']:
            imgpath = annot['name']
            img = Image.open(imgpath)
            w, h = img.size
            self.gtdict[annot['image id']] = {
                'name': imgpath.split('/')[-1],
                'size': (h, w),
                'annPoints': np.array(annot['locations']).reshape(-1,2)
            }
        

    def get_num_samples(self):
        return self.num_samples

    def gen_dot(self):
        for imgid in self.gtdict.keys():
            points = np.array(self.gtdict[imgid]['annPoints'])
            imshape = self.gtdict[imgid]['size']
            im_dot = np.zeros(imshape)
            
            for j in range(points.shape[0]):
                x = int(points[j][1])
                y = int(points[j][0])

                if x >= imshape[0] or x < 0 or y >= imshape[1] or y < 0:
                    continue
                
                im_dot[x, y] += 1.0
            self.dot_list.append(im_dot)

    def gen_den(self, sigma_default = 3, adapt = False):
        for imgid in self.gtdict.keys():
            imgname = self.gtdict[imgid]['name']
            points = np.array(self.gtdict[imgid]['annPoints'])
            imshape = self.gtdict[imgid]['size']
            im_density = np.zeros(imshape)
            if adapt:
                tree = KDTree(points.copy(), leafsize=2048)
                distances, locations = tree.query(points, k=4)
            
            # print("generating density map...")
            for j in range(points.shape[0]):
                x = int(points[j][1])
                y = int(points[j][0])
                if x >= imshape[0] or x < 0 or y >= imshape[1] or y < 0:
                    continue

                pt2d = np.zeros(imshape)
                pt2d[x, y] = 1.0
                if adapt:
                    if points.shape[0] >= 3:
                        sigma = (distances[j][1] + distances[j][2] + distances[j][3]) * 0.1
                    else:
                        sigma = np.average(distances[j][:points.shape[0]+1]) * 0.3
                else:
                    sigma = sigma_default

                im_density = im_density + gaussian_filter(pt2d, sigma, mode='constant')

            self.den_list.append(im_density)
            print("image: " + imgname +  " count: " + str(points.shape[0]) + "  den: " + str(np.sum(im_density)))
