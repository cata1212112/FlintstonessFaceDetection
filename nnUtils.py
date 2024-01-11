import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv

DATA_MEAN = (0.5, 0.5, 0.5)
DATA_STD = (0.5, 0.5, 0.5)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DATA_MEAN, std=DATA_STD)
])

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index]['Image']
        if img_path in self.cache:
            img = self.cache[img_path]
        else:
            img = cv.imread(img_path)
            self.cache[img_path] = img
        img = self.transform(img)
        label = self.data.iloc[index]['Class']
        return img, np.float32(label)

class Trainer:
    def __init__(self, network, train_loader, validation_loader, EPOCHS):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.EPOCHS = EPOCHS
        self.optimizer = optim.Adam(network.parameters(), lr=0.001)
        self.loss_function = nn.BCEWithLogitsLoss()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    def train_one_epoch(self, epoch):
        running_loss = 0
        avg_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.network(inputs).squeeze(1)

            loss = self.loss_function(outputs, labels)
            loss.backward()

            self.optimizer.step()

            correct += ((F.sigmoid(outputs) > 0.5) == labels).float().sum()
            total += len(inputs)

            running_loss += loss.item()

            if i % 10 == 9:
                avg_loss = running_loss / 10
                print('  batch {} loss: {}'.format(i + 1, avg_loss))
                self.writer.add_scalar('Loss/train', avg_loss, epoch * len(self.train_loader) + 1)
                running_loss = 0

        accuracy = 100 * correct / total
        print("Accuracy = {}".format(accuracy))
        return avg_loss

    def train(self):
        best_val_loss = 1_000_000.
        for epoch in range(self.EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            self.network.train(True)
            avg_loss = self.train_one_epoch(epoch)

            running_validation_loss = 0

            self.network.eval()

            with torch.no_grad():
                for i, validation_data in enumerate(self.validation_loader):
                    val_inputs, val_labels = validation_data

                    val_inputs = val_inputs.to(self.device)
                    val_labels = val_labels.to(self.device)

                    val_outputs = self.network(val_inputs).squeeze(1)
                    validation_loss = self.loss_function(val_outputs, val_labels)
                    running_validation_loss += validation_loss.item()

            avg_val_loss = running_validation_loss / len(self.validation_loader)
            print('LOSS train {} valid {}'.format(avg_loss, avg_val_loss))

            self.writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_val_loss},
                               epoch + 1)
            self.writer.flush()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = "models/model_cnn.pth"
                torch.save(self.network.state_dict(), model_path)

def get_train_val_loader():
    positives = "pozitive"
    negatives = "negative"

    poz = []
    neg = []

    for file in os.listdir(positives):
        if file[-3:] == "jpg":
            poz.append(os.path.join(positives, file))

    for file in os.listdir(negatives):
        if file[-3:] == "jpg":
            neg.append(os.path.join(negatives, file))

    random.shuffle(poz)
    random.shuffle(neg)

    keep = min(len(poz), len(neg))
    poz = poz[:keep]
    neg = neg[:keep]

    data = {'Image': poz + neg, 'Class': [1] * keep + [0] * keep}
    df = pd.DataFrame(data)

    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    train_size = 0.7
    train_df, validation_df = train_test_split(shuffled_df, train_size=train_size, stratify=shuffled_df['Class'])

    dataset_train = CustomDataset(train_df, transform)
    train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)

    dataset_val = CustomDataset(validation_df, transform)
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=True)

    return train_loader, val_loader
