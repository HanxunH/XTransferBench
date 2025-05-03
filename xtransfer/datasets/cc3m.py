import os
import h5py
import csv
import numpy as np
import copy
from PIL import Image
from open_clip import get_tokenizer
from torch.utils.data import Dataset


class ConceptualCaptionsDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, tokenizer=None, 
                 safe_mode_override=False, **kwargs):
                
        self.file_list = []
        csvfile = os.path.join(root, 'data/data.csv')
        with open(csvfile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.file_list.append((row[0], row[1], row[2]))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        if tokenizer is not None:
            self.tokenizer = get_tokenizer(tokenizer)
        else:
            self.tokenizer = None
        
        if 'get_idx' in kwargs and kwargs['get_idx']:
            self.get_idx = True 
        else:
            self.get_idx = False 

        # Detect Remove index
        if 'safe_mode' in kwargs and kwargs['safe_mode'] and not safe_mode_override:
            self.safe_mode = True
            safe_precentage = kwargs['safe_precentage']
            split = int(len(self.file_list) * safe_precentage)
            # Load scores
            hf = h5py.File(kwargs['safe_idx_path'], 'r')
            idx = np.argsort(hf['data'])
            self.file_list = np.array(self.file_list)
            self.file_list = self.file_list[idx[:split]].tolist()
            hf.close()
            print('Safe mode is on, {} samples are used'.format(len(self.file_list)))
            
    def __len__(self):
        return len(self.file_list)
    
    def _get_pairs(self, index):
        text = self._get_text(index)
        image = self._get_image(index)
        return image, text 
    
    def _get_image(self, index):
        image_file = self.file_list[index][0]
        image = Image.open(os.path.join(self.root, 'data/images', image_file)).convert('RGB')
        return image

    def _get_text(self, index):
        return self.file_list[index][1] 

    def __getitem__(self, idx):
        image, text = self._get_pairs(idx)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            text = self.target_transform(text)
        if self.tokenizer is not None:
            text = self.tokenizer(text)[0]
        if self.get_idx:
            return idx, image, text
        return image, text


