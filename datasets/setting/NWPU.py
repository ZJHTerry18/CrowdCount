from easydict import EasyDict as edict


# init
__C_NWPU = edict()

cfg_data = __C_NWPU

__C_NWPU.TRAIN_SIZE = (576,768)
__C_NWPU.DATA_PATH = '/data/zhaojiahe/datasets/NWPU-Crowd/'
__C_NWPU.GT_MODE = 'den'

__C_NWPU.MEAN_STD = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])

__C_NWPU.LABEL_FACTOR = 1
__C_NWPU.LOG_PARA = 100.

__C_NWPU.RESUME_MODEL = ''#model path
__C_NWPU.TRAIN_BATCH_SIZE = 8 #imgs

__C_NWPU.VAL_BATCH_SIZE = 1 # must be 1




