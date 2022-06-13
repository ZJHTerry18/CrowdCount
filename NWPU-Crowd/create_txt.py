import os
import os.path as osp
from PIL import Image
from loguru import logger
from tqdm import tqdm
from datasets.setting.NWPU import cfg_data

train_size = 0.7

root_path = cfg_data.DATA_PATH
train_path = osp.join(root_path, 'testImages')
test_path = osp.join(root_path, 'testImages')

trainvalimgfiles = os.listdir(train_path)
train_files = trainvalimgfiles[:int(len(trainvalimgfiles) * train_size)]
val_files = trainvalimgfiles[int(len(trainvalimgfiles) * train_size):]
test_files = os.listdir(test_path)

logger.info('creating train txt file...')
with open(osp.join(root_path, 'train.txt'), 'w') as f:
    lines = []
    for imgfile in tqdm(train_files):
        try:
            img = Image.open(osp.join(train_path, imgfile))
            img.load()
            img.close()
        except OSError:
            print("trainImage truncated: {}".format(imgfile))
            continue

        line = imgfile + ' 1 1\n'
        lines.append(line)
    f.writelines(lines)

logger.info('creating validation txt file...')
with open(osp.join(root_path, 'val.txt'), 'w') as f:
    lines = []
    for imgfile in tqdm(val_files):
        try:
            img = Image.open(osp.join(train_path, imgfile))
            img.load()
            img.close()
        except OSError:
            print("trainImage truncated: {}".format(imgfile))
            continue
        line = imgfile + ' 1 1\n'
        lines.append(line)
    f.writelines(lines)

logger.info('creating test txt file...')
with open(osp.join(root_path, 'test.txt'), 'w') as f:
    lines = []
    for imgfile in tqdm(test_files):
        try:
            img = Image.open(osp.join(test_path, imgfile))
            img.load()
            img.close()
        except OSError:
            print("testImage truncated: {}".format(imgfile))
            continue
        line = imgfile + '\n'
        lines.append(line)
    f.writelines(lines)