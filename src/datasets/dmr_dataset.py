import sys
from glob import glob
from typing import Tuple

import numpy as np
import torch.nn.functional
from PIL import Image
from torch.utils.data import Dataset
from json import load
import os
import cv2


RED = np.array([255,0,0])
GREEN = np.array([0,255,0])
BLUE=np.array([0,0,255])
YELLOW = RED+GREEN
CYAN = GREEN+BLUE
PINK = RED+BLUE
BLACK = np.zeros(3)
SKIN = RED+GREEN*0.8+BLUE*0.65

ID_TO_COLORS = {1:RED,2:GREEN,3:BLUE,4:YELLOW,5:CYAN,6:PINK,7:BLACK,8:SKIN}


class DMRDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/',
                 image_folder: str = '0001'):

        # Load file paths.
        self.data_image = []
        folders = glob('%s/*' % (base_path))

        for folder in folders:
            self.img_path = glob('%s/*' % (folder))

            self.imgs = sorted([img for img in self.img_path if ".jpg" in img and "snap" in img])



            for img in self.imgs:
                cur_img = np.array(Image.open(img))
                if cur_img.shape == (120, 160, 3):
                    self.data_image.append(cur_img)

        self.data_image = np.array(self.data_image)

        self.data_image = (self.data_image / 255 * 2) - 1

        # channel last to channel first to comply with Torch.

        self.data_image = np.moveaxis(self.data_image, -1, 1)

    def __len__(self) -> int:
        return self.data_image.shape[0]

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        return image, np.nan

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]

    def num_classes(self) -> int:
        return None


class DMR_gray(Dataset):

        def __init__(self,
                     base_path: str = '../../data/',
                     image_folder: str = '0001'):

            # Load file paths.
            self.data_image = []
            colored_folders = 0
            folders = glob('%s/*' % (base_path))

            for folder in folders:

                colored = 0
                self.img_path = glob('%s/*' % (folder))

                self.imgs = sorted([img for img in self.img_path if ".jpg" in img and not "snap" in img])

                for img in self.imgs:
                    cur_img = np.array(Image.open(img))

                    if not np.all(cur_img[:, :, 0] == cur_img[:, :, 1]) & np.all(cur_img[:, :, 1] == cur_img[:, :, 2]):
                        colored += 1
                        continue
                    if cur_img.shape != (480,640,3):
                        continue

                    self.data_image.append(cur_img[...,0])

                if colored > 1:
                    colored_folders += 1
            self.data_image = np.array(self.data_image)

            self.data_image = (self.data_image / 255 * 2) - 1

            # channel last to channel first to comply with Torch.

            self.data_image = np.expand_dims(np.moveaxis(self.data_image, -1, 1), 1)

        def __len__(self) -> int:
            return self.data_image.shape[0]

        def __getitem__(self, idx) -> Tuple[np.array, np.array]:
            image = self.data_image[idx]

            return image, np.nan

        def all_images(self) -> np.array:
            return self.data_image

        def num_image_channel(self) -> int:
            # [B, C, H, W]
            return self.data_image.shape[1]

        def num_classes(self) -> int:
            return None


class DMRDatasetAnnotated(Dataset):

    def __init__(self,
                 base_path: str = '../../data/',
                 image_folder: str = '0001'):

        # Load file paths.
        annotation_path = os.path.join(base_path,"52_instances_default.json")
        with open(annotation_path,'r') as f:
            antns = load(f)


        self.data_image = []
        self.data_labels = []
        self.full_paths = []

        annot_id = 0
        non_ROI_category = -1
        while annot_id < len(antns['annotations']):
            cur_img_id = antns['annotations'][annot_id]['image_id']
            image_filename = antns['images'][cur_img_id-1]['file_name']
            folder = image_filename[1:5]
            self.image_path = full_path = os.path.join(base_path,folder,image_filename)
            raw_label = np.array(Image.open(full_path))
            one_hot_label = np.zeros((raw_label.shape[0], raw_label.shape[1], 8), dtype=int)
            cur_img = np.array(Image.open(full_path.replace(".jpg","_snap.jpg")))
            if cur_img.shape == (120, 160, 3) and raw_label.shape == (480, 640, 3):
                self.data_image.append(cur_img)
                self.full_paths.append(full_path)
            else:
                annot_id += 1
                continue

            while annot_id < len(antns['annotations']) and antns['annotations'][annot_id]['image_id'] == cur_img_id:

                mask = np.zeros((cur_img.shape[0],cur_img.shape[1]), dtype=np.uint8)
                # Define the polygon coordinates
                polygon = np.array(antns['annotations'][annot_id]['segmentation'][0]).reshape(-1, 2)
                # Convert the polygon to a numpy array
                polygon_np = np.array(polygon, np.int32)

                # Reshape the polygon array into the required format by OpenCV
                polygon_np = polygon_np.reshape((-1, 1, 2))
                mask = np.kron(mask, np.ones((4, 4), dtype=mask.dtype))  # to match to label image dims
                # Draw the polygon on the mask
                cv2.fillPoly(mask, [polygon_np], 255)

                if antns['categories'][antns['annotations'][annot_id]['category_id'] - 1]['name'] != 'ROI':
                    one_hot_label[mask != 0] = np.zeros(8)  # if a label already exists, run it over
                    one_hot_label[mask != 0, antns['annotations'][annot_id]['category_id'] - 1] = 1
                else:
                    non_ROI_category = antns['annotations'][annot_id]['category_id'] - 1




                annot_id += 1

            #Give non-ROI label if there are any remaining unlabeled pixels
            one_hot_label[np.sum(one_hot_label,axis=-1)==0,non_ROI_category] =1
            self.data_labels.append(one_hot_label)


        self.data_image = np.array(self.data_image)
        self.data_labels = np.array(self.data_labels)

        self.data_image = (self.data_image / 255 * 2) - 1


        self.data_image = np.moveaxis(self.data_image, -1, 1)
        self.data_labels = np.moveaxis(self.data_labels, -1, 1)

    def __len__(self) -> int:
        return self.data_image.shape[0]

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        label = self.data_labels[idx]
        return image, label, self.full_paths[idx]

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]

    def num_classes(self) -> int:
        return 8


class DMRDatasetAnnotatedGray(Dataset):

    def __init__(self,
                 base_path: str = '../../data/retina',
                 image_folder: str = '0001'):

        # Load file paths.
        annotation_path = os.path.join(base_path,"52_instances_default.json")
        with open(annotation_path,'r') as f:
            antns = load(f)





        self.data_image = []
        self.data_labels = []
        self.full_paths = []


        annot_id = 0
        non_ROI_category = -1
        while annot_id < len(antns['annotations']):
            cur_img_id = antns['annotations'][annot_id]['image_id']
            image_filename = antns['images'][cur_img_id-1]['file_name']
            folder = image_filename[1:5]
            self.image_path = full_path = os.path.join(base_path,folder,image_filename)
            raw_label = np.array(Image.open(full_path))
            one_hot_label = np.zeros((raw_label.shape[0], raw_label.shape[1], 8), dtype=int)
            cur_img = raw_label
            if  raw_label.shape == (480, 640, 3):
                self.data_image.append(np.expand_dims(cur_img[...,1],axis=-1))
                self.full_paths.append(full_path)
            else:
                annot_id += 1
                continue

            while annot_id < len(antns['annotations']) and antns['annotations'][annot_id]['image_id'] == cur_img_id:

                mask = np.zeros((cur_img.shape[0],cur_img.shape[1]), dtype=np.uint8)
                # Define the polygon coordinates
                polygon = np.array(antns['annotations'][annot_id]['segmentation'][0]).reshape(-1, 2)
                # Convert the polygon to a numpy array
                polygon_np = np.array(polygon, np.int32)

                # Reshape the polygon array into the required format by OpenCV
                polygon_np = polygon_np.reshape((-1, 1, 2))

                # Draw the polygon on the mask
                cv2.fillPoly(mask, [polygon_np], 255)


                if antns['categories'][antns['annotations'][annot_id]['category_id'] - 1]['name'] != 'ROI':
                    
                    one_hot_label[mask != 0] = np.zeros(8)  # if a label already exists, run it over
                    one_hot_label[mask != 0, antns['annotations'][annot_id]['category_id'] - 1] = 1
                else:
                    non_ROI_category = antns['annotations'][annot_id]['category_id'] - 1
                



                annot_id += 1

            #Give non-ROI label if there are any remaining unlabeled pixels
            one_hot_label[np.sum(one_hot_label,axis=-1)==0,non_ROI_category] =1
            self.data_labels.append(one_hot_label)


      
        self.data_image = np.array(self.data_image)
        self.data_labels = np.array(self.data_labels)

        self.data_image = (self.data_image / 255 * 2) - 1
       


        self.data_image = np.moveaxis(self.data_image, -1, 1)
        self.data_labels = np.moveaxis(self.data_labels, -1, 1)

    def __len__(self) -> int:
        return self.data_image.shape[0]

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.data_image[idx]
        label = self.data_labels[idx]

        return image, label, self.full_paths[idx]

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return self.data_image.shape[1]

    def num_classes(self) -> int:
        return 8

class DMRDatasetClassificationGray(Dataset):
    def __init__(self,
                 base_path: str = '../../data/',
                 image_folder: str = '0001'):
        self.data_image = {}
        self.data_labels = {}
        self.folders = []
        self.full_paths = {}
        
        
        for folder in os.listdir(base_path):
           
            if folder not in self.data_image:
                self.data_image[folder] = []
            if folder not in self.full_paths:
                self.full_paths[folder] = []

            if not os.path.isdir(os.path.join(base_path,folder)):
                del self.data_image[folder]
                del self.full_paths[folder]
                continue
            if "description.html" not in os.listdir(os.path.join(base_path,folder)):

                del self.data_image[folder]
                del self.full_paths[folder]
                continue
            images = [x for x in os.listdir(os.path.join(base_path,folder)) if x.endswith(".jpg") and not x.endswith("snap.jpg")]
            descriptor_path = os.path.join(base_path,folder,"description.html")
            with open(descriptor_path,"r") as f:
                html_data = f.read()
            label_str = html_data.split("Diagnosis*: <span>")[1].split("<br>")[0]
            if label_str not in ["Healthy","Sick"]:

                del self.data_image[folder]
                del self.full_paths[folder]
                continue

            self.data_labels[folder] = 0 if label_str == "Healthy" else 1
            self.folders.append(folder)

            for image in images:
                full_path = os.path.join(base_path,folder,image)
                cur_img = np.expand_dims(np.array(Image.open(full_path))[...,1],axis=-1)
                if cur_img.shape == (480, 640, 1):
                    
                    data_image = np.expand_dims(np.array(cur_img),0)
                    data_image = np.moveaxis((data_image / 255 * 2) - 1,-1,1)
                    self.data_image[folder].append(data_image)
                    self.full_paths[folder].append(full_path)
                else:

                    continue

            if self.data_image[folder] == []:
                del self.data_image[folder]
                del self.full_paths[folder]
                continue
            if folder not in self.data_image:
                continue
         
            self.data_image[folder] = np.array(self.data_image[folder]).squeeze(1)
            self.data_labels[folder] = np.array(self.data_labels[folder])



        all_folders = sorted(self.data_image.keys(), key=lambda key: self.data_image[key].shape[0], reverse=True)
        sizes = {key:self.data_image[key].shape[0] for key in all_folders}
        train_size = sum(sizes.values())
        test_size = 0
        ratio = 0.8
        self.test_folders = []
        self.train_folders = all_folders.copy()
        while(train_size/(train_size+test_size)>0.8):

            self.test_folders.append(all_folders[0])
            self.test_folders.append(all_folders[-1])
            self.train_folders.remove(all_folders[0])
            self.train_folders.remove(all_folders[-1])
            train_size = train_size - sizes[all_folders[0]] - sizes[all_folders[-1]]
            test_size = test_size + sizes[all_folders[0]] + sizes[all_folders[-1]]
            all_folders.remove(all_folders[0])
            all_folders.remove(all_folders[-1])

        self.train_images = None
        self.test_images = None
        self.train_labels = []
        self.test_labels = []

        for folder in self.train_folders:
            if self.train_images is None:
                self.train_images = self.data_image[folder]

            else:
                self.train_images = np.concatenate((self.train_images,self.data_image[folder]))
            self.train_labels += [self.data_labels[folder].item()]*self.data_image[folder].shape[0]


        for folder in self.test_folders:
            if self.test_images is None:
                self.test_images = self.data_image[folder]

            else:
                self.test_images = np.concatenate((self.test_images,self.data_image[folder]))
            self.test_labels += [self.data_labels[folder].item()]*self.data_image[folder].shape[0]




    def __len__(self) -> int:
        return len(self.train_images)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.train_images[idx]
        label = self.train_labels[idx]

        return image, label

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]
        return list(self.data_image.values())[0].shape[1]

class DMRDatasetClassification(Dataset):

    def __init__(self,
                 base_path: str = '../../data/',
                 image_folder: str = '0001'):
        self.data_image = {}
        self.data_labels = {}
        self.folders = []
        self.full_paths = {}
        for folder in os.listdir(base_path):

            if folder not in self.data_image:
                self.data_image[folder] = []
            if folder not in self.full_paths:
                self.full_paths[folder] = []

            if not os.path.isdir(os.path.join(base_path,folder)):
                del self.data_image[folder]
                del self.full_paths[folder]
                continue
            if "description.html" not in os.listdir(os.path.join(base_path,folder)):

                del self.data_image[folder]
                del self.full_paths[folder]
                continue
            images = [x for x in os.listdir(os.path.join(base_path,folder)) if x.endswith("snap.jpg")]
            descriptor_path = os.path.join(base_path,folder,"description.html")
            with open(descriptor_path,"r") as f:
                html_data = f.read()
            label_str = html_data.split("Diagnosis*: <span>")[1].split("<br>")[0]
            if label_str not in ["Healthy","Sick"]:

                del self.data_image[folder]
                del self.full_paths[folder]
                continue

            self.data_labels[folder] = 0 if label_str == "Healthy" else 1
            self.folders.append(folder)

            for image in images:
                full_path = os.path.join(base_path,folder,image)
                cur_img = np.array(Image.open(full_path))
                if cur_img.shape == (120, 160, 3):
                    data_image = np.expand_dims(np.array(cur_img),0)
                    data_image = np.moveaxis((data_image / 255 * 2) - 1,-1,1)
                    self.data_image[folder].append(data_image)
                    self.full_paths[folder].append(full_path)
                else:
                    del self.data_image[folder]
                    del self.full_paths[folder]
                    continue

            self.data_image[folder] = np.array(self.data_image[folder]).squeeze()
            self.data_labels[folder] = np.array(self.data_labels[folder])



        all_folders = sorted(self.data_image.keys(), key=lambda key: self.data_image[key].shape[0], reverse=True)
        sizes = {key:self.data_image[key].shape[0] for key in all_folders}
        train_size = sum(sizes.values())
        test_size = 0
        ratio = 0.8
        self.test_folders = []
        self.train_folders = all_folders.copy()
        while(train_size/(train_size+test_size)>0.8):

            self.test_folders.append(all_folders[0])
            self.test_folders.append(all_folders[-1])
            self.train_folders.remove(all_folders[0])
            self.train_folders.remove(all_folders[-1])
            train_size = train_size - sizes[all_folders[0]] - sizes[all_folders[-1]]
            test_size = test_size + sizes[all_folders[0]] + sizes[all_folders[-1]]
            all_folders.remove(all_folders[0])
            all_folders.remove(all_folders[-1])

        self.train_images = None
        self.test_images = None
        self.train_labels = []
        self.test_labels = []


        for folder in self.train_folders:
            if self.train_images is None:
                self.train_images = self.data_image[folder]

            else:
                self.train_images = np.concatenate((self.train_images,self.data_image[folder]))
            self.train_labels += [self.data_labels[folder].item()]*self.data_image[folder].shape[0]


        for folder in self.test_folders:
            if self.test_images is None:
                self.test_images = self.data_image[folder]

            else:
                self.test_images = np.concatenate((self.test_images,self.data_image[folder]))
            self.test_labels += [self.data_labels[folder].item()]*self.data_image[folder].shape[0]




    def __len__(self) -> int:
        return len(self.train_images)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = self.train_images[idx]
        label = self.train_labels[idx]

        return image, label

    def all_images(self) -> np.array:
        return self.data_image

    def num_image_channel(self) -> int:
        # [B, C, H, W]

        return list(self.data_image.values())[0].shape[1]
        
