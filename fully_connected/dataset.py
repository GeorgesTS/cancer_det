
from torch.utils.data import Dataset
from PIL import Image
import numpy as np



class Data(Dataset):
    
    

    def __init__(self,path,labels):
        self.path=path
        self.labels=labels
        

    def __len__(self):
        return (len(self.path))


    def __getitem__(self,index):
        base_path="/home/jorje/cancer/fully_connected/data/"
        image=Image.open(base_path+self.path[index]+".pgm")
        image = np.transpose(image).astype(np.float32)
        label=self.labels[index]
        return image, label