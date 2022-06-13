# -*- coding: utf-8 -*-

import enum
import torch.utils.data as data
from . import common
from tqdm import tqdm
from loguru import logger
from config import cfg
from .setting.NWPU import cfg_data
import os
import gc
import json
from PIL import Image
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter


class NWPUDataset(data.Dataset):
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.data_path = data_path
        self.datasetname = datasetname

        self.file_name = []
        self.info = []
        self.dot_list = []
        self.den_list = []
        
        with open(argv['list_file']) as f:
            lines = f.readlines()

        self.gtmode = argv['gt_mode']
        self.imgfileTemp = os.path.join(data_path, 'images', '{}.jpg')
        self.matfileTemp = os.path.join(data_path, 'mats', '{}.mat')
        self.dotfileTemp = os.path.join(data_path, self.gtmode, '{}.png')
        if self.gtmode == 'dot':
            logger.info('generating dot map...')
            self.gen_dot()
        # elif self.gtmode == 'den':
        #     logger.info('generating density map...')
        #     self.gen_den()
        #     pass

        for line in lines:
            splited = line.strip().split()
            filename = splited[0]
            img = Image.open(self.imgfileTemp.format(filename)).convert('RGB')
            w, h = img.size
            img.close()
            if 1080 >= h >= cfg_data.TRAIN_SIZE[0] and 1920 >= w >= cfg_data.TRAIN_SIZE[1]:
                self.file_name.append(splited[0])
                self.info.append(splited[1:3]) # lum, crowd level

        self.num_samples = len(self.file_name)
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
        imshape = img.size

        if self.gtmode == 'dot':
            gt = self.dot_list[index]
        elif self.gtmode == 'den':
            # gt = self.den_list[index]
            gt = self.gen_den(index, sigma_default=cfg.GS_SIGMA, adapt=False)
        gt = Image.fromarray(gt)
      
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
        img_path = self.imgfileTemp.format(self.file_name[index])
        img = Image.open(img_path).convert('RGB')

        return img

    def read_gt(self,index):
        gt_path = self.dotfileTemp.format(self.file_name[index])
        gt = Image.open(gt_path)

        return gt
    
    def read_mat(self,index):
        mat_path = self.matfileTemp.format(self.file_name[index])
        mat_data = loadmat(mat_path)
        ann = mat_data['annPoints']

        return ann

    def get_num_samples(self):
        return self.num_samples

    def gen_dot(self):
        for index, imgname in tqdm(enumerate(self.file_name)):
            img = np.array(self.read_image(index))
            imshape = img.shape[:2]
            im_dot = np.zeros(imshape)
            points = self.read_mat(index)
            
            for j in range(points.shape[0]):
                x = int(points[j][1])
                y = int(points[j][0])

                if x >= imshape[0] or x < 0 or y >= imshape[1] or y < 0:
                    continue
                
                im_dot[x][y] += 1.0
            self.dot_list.append(im_dot)
    
    def gen_den(self, index, sigma_default = 3, adapt = False):
        imgname = self.file_name[index]
        img = np.array(self.read_image(index))
        imshape = img.shape[:2]
        im_density = np.zeros(imshape)
        points = self.read_mat(index)

        if adapt:
            tree = KDTree(points.copy(), leafsize=2048)
            distances, locations = tree.query(points, k=4)
        
        # print("generating density map...")
        sigma = sigma_default
        for j in range(points.shape[0]):
            x = int(points[j][1])
            y = int(points[j][0])
            if x >= imshape[0] or x < 0 or y >= imshape[1] or y < 0:
                continue

            im_density[x, y] = 1.0
            if adapt:
                if points.shape[0] >= 4:
                    sigma = (distances[j][1] + distances[j][2] + distances[j][3]) * 0.1
                else:
                    sigma = np.average(distances[j][1:points.shape[0]]) * 0.3
            else:
                sigma = sigma_default

        im_density = gaussian_filter(im_density, sigma, mode='constant')
        # print("image: " + imgname +  " count: " + str(points.shape[0]) + "  den: " + str(np.sum(im_density)))
        return im_density


class JingtouTrainDataset(data.Dataset):
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.data_path = data_path
        self.datasetname = datasetname

        self.file_id = []
        self.info = []
        self.gtdict = {}
        self.dot_list = []
        self.den_list = []
        self.img_size = cfg_data.TRAIN_SIZE

        self.gtmode = argv['gt_mode']
        self.list_file = argv['list_file']

        with open(self.list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                self.info.append(splited[1:3]) # lum, crowd level

        logger.info('reading annotation file...')
        self.read_annot(os.path.join(data_path, 'annotations', 'test.json')) # self.gtdict: {'000001': {'name':'05-zhaji-1', 'size':(1080,1920),'ann':[[0.0,1.0],[1.0,2.0]...]}}
        self.file_id = list(self.gtdict.keys())
        self.den_list = [None for _ in range(len(self.file_id))]

        self.imgfileTemp = os.path.join(data_path, 'testImages', '{}')
        # self.matfileTemp = os.path.join(data_path, 'mats', '{}.mat')
        # self.gtfileTemp = os.path.join(data_path, self.gtmode, '{}.png')
        if self.gtmode == 'dot':
            logger.info('generating dot map...')
            self.gen_dot()
        # elif self.gtmode == 'den':
        #     logger.info('generating density map...')
        #     self.gen_den(sigma_default=10, adapt=False)

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
            print(f'[Jingtou DATASET]: {self.num_samples} training images.')
        if self.mode == 'val':
            print(f'[Jingtou DATASET]: {self.num_samples} validation images.')   

    
    def __getitem__(self, index):
        img = self.read_image(index)
        # gt = self.read_gt(index)
        
        if self.gtmode == 'dot':
            gt = self.dot_list[index]
        elif self.gtmode == 'den':
            if self.den_list[index] != None:
                gt = self.den_list[index]
            else:
                gt = self.gen_den(index, sigma_default=cfg.GS_SIGMA, adapt=False)
            # gt = self.den_list[index]
        gt = Image.fromarray(gt)

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
        img_path = self.imgfileTemp.format(self.gtdict[self.file_id[index]]['name'])
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

        for iminfo in tqdm(dat['images']):
            if iminfo['file_name'] in data_names:
                h = iminfo['height']
                w = iminfo['width']
                if h >= self.img_size[0] and w >= self.img_size[1]:
                    self.gtdict[iminfo['id']] = {
                        'name': iminfo['file_name'],
                        'size': (iminfo['height'], iminfo['width']),
                        'annPoints': []
                    }
        
        for annot in dat['annotations']:
            if annot['image_id'] in self.gtdict.keys():
                self.gtdict[annot['image_id']]['annPoints'].append(annot['headPoint'])


    def read_cr_annot(self, json_path):
        with open(self.list_file, 'r') as f:
            data_names = f.readlines()
        data_names = [x.split()[0] for x in data_names]

        with open(json_path, 'r') as f:
            dat = json.load(f)
        
        for annot in tqdm(dat['annotations']):
            imgpath = annot['name']
            imgname = imgpath.split('/')[-1]
            if imgname in data_names:
                img = Image.open(os.path.join(self.data_path, imgpath))
                w, h = img.size
                if h >= self.img_size[0] and w >= self.img_size[1]:
                    self.gtdict[annot['image id']] = {
                        'name': imgname,
                        'size': (h, w),
                        'annPoints': np.array(annot['locations']).reshape(-1,2)
                    }
                img.close()
        

    def get_num_samples(self):
        return self.num_samples

    def gen_dot(self):
        for imgid in tqdm(self.gtdict.keys()):
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

    def gen_den(self, index, sigma_default = 3, adapt = False):
        def gauss_map(shape, center, sigma):
            x, y = np.meshgrid(range(shape[1]), range(shape[0]))
            x0 = center[0]
            y0 = center[1]

            output = 1.0 / (2 * np.pi * (sigma ** 2)) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) * 0.5 / sigma ** 2)
            return output         

        imgid = self.file_id[index]
        imgname = self.gtdict[imgid]['name']
        points = np.array(self.gtdict[imgid]['annPoints'])
        imshape = self.gtdict[imgid]['size']
        im_density = np.zeros(imshape)
        if adapt:
            tree = KDTree(points.copy(), leafsize=2048)
            distances, locations = tree.query(points, k=4)
        
        # pt2d = np.zeros(imshape)
        sigma = sigma_default
        for j in range(points.shape[0]):
            x = int(points[j][1])
            y = int(points[j][0])
            if x >= imshape[0] or x < 0 or y >= imshape[1] or y < 0:
                continue
            
            # pt2d[pt2d > 0] = 0.0
            # pt2d[x, y] = 1.0
            im_density[x, y] = 1.0
            if adapt:
                if points.shape[0] >= 3:
                    sigma = (distances[j][1] + distances[j][2] + distances[j][3]) * 0.1
                else:
                    sigma = np.average(distances[j][:points.shape[0]+1]) * 0.3
            else:
                sigma = sigma_default

        im_density = gaussian_filter(im_density, sigma)
        # self.den_list[index] = im_density
        # im_density = gauss_map(imshape, (x,y), sigma)
        # print("image: " + imgname +  " count: " + str(points.shape[0]) + "  den: " + str(np.sum(im_density)))
        return im_density