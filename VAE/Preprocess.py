# Do some preprocessing about the input data
# Input data is in npy format
import torch
import numpy as np
import json

# should in 28 * 28
RAW_BITMAP_CAT_PATH = '/Users/dingfan/FinalYearProject/VAE/Data/bitmap_cat.npy'
BITMAP_CAT_TRAIN_PATH = '/Users/dingfan/FinalYearProject/VAE/Data/cat_train_set'
BITMAP_CAT_TEST_PATH = '/Users/dingfan/FinalYearProject/VAE/Data/cat_test_set'

NUM_TRAIN = 5000
NUM_TEST = 5000

def preprocess():
    bitmap = np.load(RAW_BITMAP_CAT_PATH)
    # chosen_bitmap = []

    torch.save(bitmap[:NUM_TRAIN], open(BITMAP_CAT_TRAIN_PATH, 'wb'))
    torch.save(bitmap[NUM_TRAIN:NUM_TRAIN+NUM_TEST], open(BITMAP_CAT_TEST_PATH, 'wb'))

preprocess()







