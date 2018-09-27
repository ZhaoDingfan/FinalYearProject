import torch
import numpy as np
import os

from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

class VAEInput(Dataset):
    def __init__(self, data_folder, mode, transform=None):
        self.transform = transform
        data_files = os.listdir(data_folder)
        data_filenames = {}
        # for data_file in data_files:
        #     splitted = data_file.split('_')
        #     if splitted[1] == mode + '.pt':
        #         data_filenames[splitted[0]] = data_file
        
        # self.drawings = []
        # for category, file_name in data_filenames.items():
        #     drawings = torch.load(os.path.join(data_folder, file_name))
        #     for drawing in drawings:
        #         self.drawings.append((drawing, category))
        
        for data_file in data_files:
            splitted = data_file.split('_')
            if splitted[1] == mode:
                # make it under category of cat
                data_filenames[splitted[0]] = data_file
        
        self.drawings = []
        for category, file_name in data_filenames.items():
            drawings = torch.load(os.path.join(data_folder, file_name))
            for drawing in drawings:
                self.drawings.append((drawing, category))
    
    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, idx): 
        return torch.tensor(
            np.reshape(np.multiply(self.drawings[idx][0], 1./255), (1, 28, 28)), 
            dtype=torch.float
            ), self.drawings[idx][1]
