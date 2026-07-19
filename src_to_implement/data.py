from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from skimage.color import gray2rgb
import cv2
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self, data, mode: str):
        super().__init__()

        self.data = data
        self.mode = mode ### val or train
        
        self.image_path = ''
        for root, _, files in os.walk('.'):
            for name in files:
                if name == 'images.zip':
                    self.image_path = os.path.join(root, name)
        
        self._transform_train  = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            
            # ### Bild Augmentationen
          #  tv.transforms.RandomEqualize(0.1),
            # tv.transforms.RandomAffine(
            #     degrees=0,
            #     translate=(0.05,0.05),
            #     scale=(0.9,1.1)
            # ),
           # tv.transforms.RandomAutocontrast(0.1),
            tv.transforms.RandomVerticalFlip(0.3),
            tv.transforms.RandomHorizontalFlip(0.3),
            tv.transforms.RandomApply([
                 tv.transforms.GaussianBlur(kernel_size=5,
                                            sigma=(0.1,0.5))
                
            ],p=0.2),
          #  tv.transforms.RandomInvert(0.2),
            
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=train_mean,std=train_std)
            ])
        
        self._transform_test  = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=train_mean,std=train_std)
        ])

     
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #print(f"[DEBUG] {index}")
        
        sample = self.data.iloc[[index]].values
        path = sample[0,0]
        
        crack = sample[0,1]
        inactive = sample[0,2]
        
        labels = np.array([crack,inactive],dtype=np.float32)
        
        # src_to_implement\images.zip
        
        img= imread(self.image_path+"/"+path)
       
        img = gray2rgb(img)
        
        
        if self.mode == "train":
            img = self._transform_train(img)
        else:
            img = self._transform_test(img)
        labels = torch.from_numpy(labels)
        
        return img, labels
        
