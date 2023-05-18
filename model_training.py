import cv2
import os
import random
import numpy as np

import itertools
import random

from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
from torch import optim

from inception_resnet_v1 import InceptionResnetV1


def preprocess_dataset(old_path, new_path):
    print('Preprocess: start..')

    os.makedirs(new_path)
    for dir_name in os.listdir(old_path):
        for file in os.listdir(os.path.join(old_path, dir_name)): 
            path = os.path.join(old_path, dir_name, file)
            image = cv2.imread(path)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face = faceCascade.detectMultiScale(gray_img, 1.2, 5) 
            if len(face) != 0:
                x, y, w, h = face[0]
                numpy_image = image[y : y + h, x : x + w]
                
                if not os.path.exists(os.path.join(new_path, dir_name)):
                    os.makedirs(os.path.join(new_path, dir_name))

                new_path_full = os.path.join(new_path, dir_name, file)
                cv2.imwrite(new_path_full, numpy_image)
    
    print('Preprocess: done!')


def get_pairs(dataset_dir):
    """
    Returns the list of the pairs: anchor, positive, negative.
    """
    all_images = []  
    for dir_name in os.listdir(dataset_dir):
        for file in os.listdir(os.path.join(dataset_dir, dir_name)): 
            all_images.append(file)
      
    pairs = []
    for dir_name in os.listdir(dataset_dir):
        files_list = os.listdir(os.path.join(dataset_dir, dir_name))

        # If directory (person) has more than 1 image
        if len(files_list) != 1: 
            # Make combinations among images of the same person
            for pair in itertools.combinations(files_list, 2): 
                while (True):
                    index = int(random.random() * (len(all_images) - 1))
                    image = all_images[index]
                    # Find negative example for the image (another person)
                    if image not in files_list:
                        pairs.append((pair[0], pair[1], image)) # anchor, positive, negative
                        break
    
    return pairs

class LFW_Dataset(Dataset):
    def __init__(self, image_folder_dataset, transform = None):
        self.image_folder_dataset = image_folder_dataset    
        self.transform = transform

        self.pairs = get_pairs(image_folder_dataset)


    def load_image(self, image_name):
        image_path = os.path.join(self.image_folder_dataset, image_name.rsplit('_', 1)[0], image_name)
        return Image.open(image_path).convert('RGB') 
        
    def __getitem__(self, index):
        pair_path = self.pairs[index] 
        pair  = [self.load_image(image_path) for image_path in pair_path]
        
        if self.transform is not None:
            pair = [self.transform(image) for image in pair]
        
        return pair 
    
    def __len__(self):
        return len(self.pairs)

def get_loader(data: LFW_Dataset, batch_size, n_train_batch, n_val_batch):
    val_size = n_val_batch * batch_size
    train_size = n_train_batch * batch_size

    data_len = len(data)
    indices = list(range(data_len))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[val_size:train_size + val_size], indices[:val_size]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    return train_loader, valid_loader


def train(model, train_loader, valid_loader, criterion, optimizer, eval_value):
    global train_loss_list, train_pos_d, train_neg_d, train_val_list, train_far_list 
    global valid_loss_list, valid_pos_d, valid_neg_d, valid_val_list, valid_far_list 

    for index, pair in enumerate(train_loader):
        model.train()
        anchor, positive, negative = pair
        optimizer.zero_grad()

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        pos_distance = distance(anchor_embedding, positive_embedding)
        neg_distance = distance(anchor_embedding, negative_embedding)

        train_loss = criterion(anchor_embedding, \
                        positive_embedding, negative_embedding)
        train_loss.backward()
        optimizer.step()

        train_val = pos_distance <= threshold
        pos_distance = torch.mean(pos_distance).item()

        train_far = neg_distance <= threshold
        neg_distance = torch.mean(neg_distance).item()
        
        train_val = torch.sum(train_val).item() / len(train_val)
        train_far = torch.sum(train_far).item() / len(train_far)

        print(f"{index} Current Train loss {train_loss.item()}")
        print(f"Current Train positive distance {pos_distance}")
        print(f"Current Train negative distance {neg_distance}\n")

        train_loss_list.append(train_loss.item())
        train_pos_d.append(pos_distance)
        train_neg_d.append(neg_distance)
        train_val_list.append(train_val)
        train_far_list.append(train_far)

        if index % eval_value == eval_value - 1:
            model.eval()
            with torch.no_grad():
                for anchor, positive, negative in valid_loader:
                    anchor_embedding = model(anchor)
                    positive_embedding = model(positive)
                    negative_embedding = model(negative)

                    pos_distance = distance(anchor_embedding, positive_embedding)
                    neg_distance = distance(anchor_embedding, negative_embedding)

                    valid_loss = criterion(anchor_embedding, \
                                        positive_embedding, negative_embedding)

                    valid_val = pos_distance <= threshold
                    pos_distance = torch.mean(pos_distance).item()

                    valid_far = neg_distance <= threshold
                    neg_distance = torch.mean(neg_distance).item()
                    
                    valid_val = torch.sum(valid_val).item() / len(valid_val)
                    valid_far = torch.sum(valid_far).item() / len(valid_far)

                    print(f"Current Valid loss {valid_loss.item()}")
                    print(f"Current Valid positive distance {pos_distance}")
                    print(f"Current Valid negative distance {neg_distance}\n")

                    valid_loss_list.append(valid_loss.item())
                    valid_pos_d.append(pos_distance)
                    valid_neg_d.append(neg_distance)
                    valid_val_list.append(valid_val)
                    valid_far_list.append(valid_far)
 
        
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() ])
    
    return transform(image)

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor() ])

    harcascadePath = "utils/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    
    if not (os.path.isdir('data/cropped_lfw')):
        preprocess_dataset('data/lfw', 'data/cropped_lfw')

    dataset = LFW_Dataset('data/cropped_lfw', transform)

    train_loader, valid_loader = get_loader(dataset, 100, 150, 10)

    model = InceptionResnetV1()
    state_dict = torch.load('utils/20180402-114759-vggface2.pt') 
    model.load_state_dict(state_dict)

    distance = nn.PairwiseDistance()
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance, margin=0.2)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    threshold = 0.75

    train_loss_list, train_pos_d, train_neg_d, train_val_list, train_far_list = [], [], [], [], []
    valid_loss_list, valid_pos_d, valid_neg_d, valid_val_list, valid_far_list = [], [], [], [], []

    eval_value = 10
    train(model, train_loader, valid_loader, criterion, optimizer, eval_value) 

    # def get_averaged(arr: list):
    #     return [np.average(arr[(i * eval_value): (i + 1) * eval_value]) \
    #             for i in range(int(len(arr) / eval_value) - 1)]

    torch.save(model, 'fine_tuned_model.pt')