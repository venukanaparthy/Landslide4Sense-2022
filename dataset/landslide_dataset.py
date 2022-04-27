import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import h5py
from PIL import Image
import albumentations as A

class LandslideDataSet(data.Dataset):
    def __init__(self, data_dir, list_path=None, list_path_augment=None, data_dir_augment=None, transform=None, max_iters=None,set='label'):
        self.list_path = list_path
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)] if not list_path is None else []
        self.img_aug_ids = [i_aug_id.strip() for i_aug_id in open(list_path_augment)] if not list_path_augment is None else []
        self.transform = transform


        #print(self.img_ids)        
        if not max_iters==None:
            if self.img_ids:
                n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
                #print(n_repeat, max_iters, len(self.img_ids))
                print("img-ids:{}".format(n_repeat))
                self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

            if self.img_aug_ids:                
                n_repeat = int(np.ceil(max_iters / len(self.img_aug_ids)))
                print("aug-ids:{}".format(n_repeat))
                self.img_aug_ids = self.img_aug_ids * n_repeat + self.img_aug_ids[:max_iters-n_repeat*len(self.img_aug_ids)]
            


        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })

            if not list_path_augment is None:
                print(self.img_aug_ids[:1])        
                for name in self.img_aug_ids:
                    img_file = data_dir_augment + name
                    label_file = data_dir_augment + name.replace('img','mask').replace('image','mask')
                    self.files.append({
                        'img': img_file,
                        'label': label_file,
                        'name': name
                    })
                print(self.files[-1])
                   
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):    
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        print("datafiles:{}".format(datafiles))
        if self.set=='labeled':
            name = datafiles['name']
            print("name:{}".format(name))
            if name[:7] == "AugData":
                #load using pytorch
                image = torch.load(datafiles['img'])
                label = torch.load(datafiles['label'])
            else:
                #load h5                    
                with h5py.File(datafiles['img'], 'r') as hf:
                    image = hf['img'][:]
                with h5py.File(datafiles['label'], 'r') as hf:
                    label = hf['mask'][:]
            
                
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            print("image.shape:{}, label.shape:{}".format(image.shape, label.shape))
            if name[:7] != "AugData": #augmented data has desried order. no need to transpose, CHANNELS x HEIGHT x WIDTH
                image = image.transpose((-1, 0, 1))
            print("transpose image.shape:{}".format(image.shape))
            size = image.shape

            #transform
            #TODO: do we need transpose before doing transformation?
            if self.transform is not None:
                aug = self.transform(image = image, mask = label)
                image = aug['image']
                label = aug['mask']
                print("transformd image shape: {}, label shape:{}".format(image.shape, label.shape))

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            return image.copy(), np.array(size), name

       
if __name__ == '__main__':
    
    transform = None# A.Compose([A.HorizontalFlip(), A.VerticalFlip(), A.RandomBrightnessContrast((0,0.5), (0, 0.5))])
    train_dataset = LandslideDataSet(data_dir='/Users/venu4461/Desktop/LandslideDetect/Landslide4Sense-2022', list_path='/Users/venu4461/Desktop/LandslideDetect/Landslide4Sense-2022/dataset/train.txt')
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,pin_memory=True)

    channels_sum,channel_squared_sum = 0,0
    num_batches = len(train_loader)
    for data,_,_,_ in train_loader:
        channels_sum += torch.mean(data,dim=[0,2,3])   
        channel_squared_sum += torch.mean(data**2,dim=[0,2,3])       

    mean = channels_sum/num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5
    print(mean,std) 
    #[-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
    #[0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
