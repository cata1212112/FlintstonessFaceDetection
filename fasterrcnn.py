import joblib
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN, backbone_utils
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import *
from PIL import Image
import os
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torchvision.transforms import v2 as T
from tqdm import tqdm
from utility import non_maximum_suppression, draw_rectangles_on_image_and_save, eval_detections

characters = ['', 'barney', 'betty', 'fred', 'wilma', 'unknown']
character_mapping = {'barney': 1, 'betty': 2, 'fred': 3, 'wilma': 4, 'unknown': 5}


def get_transform(train=False):
    transforms = []
    if train:
        transforms.append(T.RandomVerticalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


train_transforms = get_transform(train=True)
validation_transforms = get_transform()


class FlintstonesDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
        self.image_names = list(data.keys())

    def __getitem__(self, item):
        image_name = self.image_names[item]
        image = Image.open(image_name)
        image = image.convert("RGB")

        boxes = []
        labels = []
        for (xmin, ymin, xmax, ymax, label) in self.data[image_name]:
            labels.append(label)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([item])
        target["image_id"] = image_id

        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_names)


class MyFasterRCNN():
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 5
    CLASSES = ['barney', 'betty', 'fred', 'wilma', 'unknown']
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        custom_backbone = backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', weights=None, trainable_layers=5)
        self.model = FasterRCNN(backbone=custom_backbone, num_classes=MyFasterRCNN.NUM_CLASSES + 1, min_size=240, max_size=240)
        total_params = sum(param.numel() for param in self.model.parameters())
        print(f"{total_params:,} trainable parameters.")
        self.model.to(MyFasterRCNN.DEVICE)
        self.writer = SummaryWriter(log_dir='runs', comment='FasterRCNN')

    def load_data(self, load=False):
        if load:
            training_data = joblib.load("training_data.joblib")
            validation_data = joblib.load("validation_data.joblib")
        else:
            boxes = {}
            for character in characters:
                if character == "unknown" or character == "":
                    continue
                annotation_file = f"antrenare/{character}_annotations.txt"

                with open(annotation_file, "r") as f:
                    for line in f:
                        image_name, xmin, ymin, xmax, ymax, ch = line.split()
                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)

                        image_path = f"antrenare/{character}/{image_name}"

                        if image_path not in boxes:
                            boxes[image_path] = [(xmin, ymin, xmax, ymax, character_mapping[ch])]
                        else:
                            boxes[image_path].append((xmin, ymin, xmax, ymax, character_mapping[ch]))

            training_data = {}
            validation_data = {}

            for character in characters:
                if character == "unknown" or character == "":
                    continue
                # random_numbers = np.random.choice(range(1, 1001), size=200, replace=False)
                random_numbers = []
                for i in range(1, 1001):
                    image_name = f"antrenare/{character}/{i:04d}.jpg"
                    if i in random_numbers:
                        validation_data[image_name] = boxes[image_name]
                    else:
                        training_data[image_name] = boxes[image_name]

            joblib.dump(training_data, "training_data.joblib")
            joblib.dump(validation_data, "validation_data.joblib")

        print(f"Training images: {len(training_data)}")
        print(f"Validation images: {len(validation_data)}")

        return FlintstonesDataset(training_data, train_transforms), FlintstonesDataset(validation_data,
                                                                                       validation_transforms)

    def train(self):
        train_dataset, validation_dataset = self.load_data(load=False)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MyFasterRCNN.BATCH_SIZE, shuffle=True,
                                                        collate_fn=lambda x: list(zip(*x)))
        # validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=MyFasterRCNN.BATCH_SIZE,
        #                                                      shuffle=True, collate_fn=lambda x: list(zip(*x)))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=MyFasterRCNN.LEARNING_RATE)

        self.model.train()
        for epoch in range(MyFasterRCNN.EPOCHS):
            print(f"Epoch {epoch + 1}/{MyFasterRCNN.EPOCHS}")
            train_losses = []
            for images, targets in tqdm(train_data_loader):
                images = list(image.to(MyFasterRCNN.DEVICE) for image in images)
                targets = [{k: v.to(MyFasterRCNN.DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                train_losses.append(losses.item())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            self.writer.add_scalar('Train Loss', np.mean(train_losses), epoch)
            print(f"Train loss: {np.mean(train_losses)}")
            #
            # validation_losses = []
            # for images, targets in validation_data_loader:
            #     images = list(image.to(MyFasterRCNN.DEVICE) for image in images)
            #     targets = [{k: v.to(MyFasterRCNN.DEVICE) for k, v in t.items()} for t in targets]
            #
            #     with torch.no_grad():
            #         loss_dict = self.model(images, targets)
            #         losses = sum(loss for loss in loss_dict.values())
            #         validation_losses.append(losses.item())
            #
            # self.writer.add_scalar('Validation Loss', np.mean(validation_losses), epoch)
            # print(f"Validation loss: {np.mean(validation_losses)}")

            self.save()

    def test(self):
        self.model.load_state_dict(torch.load("models/fasterRCNN.pth"))
        self.model.eval()

        detections = []
        final_scores = []
        file_names = []

        barney_detections = []
        barney_detections_file_names = []
        barney_detections_scores = []
        betty_detections = []
        betty_detections_file_names = []
        betty_detections_scores = []
        fred_detections = []
        fred_detections_file_names = []
        fred_detections_scores = []
        wilma_detections = []
        wilma_detections_file_names = []
        wilma_detections_scores = []



        for image_name in os.listdir("validare/validare"):
            image = Image.open(f"validare/validare/{image_name}")
            image = image.convert("RGB")
            image_for_prediction = train_transforms(image.copy(), {})[0]
            with torch.no_grad():
                predictions = self.model([image_for_prediction.to(MyFasterRCNN.DEVICE)])
                predictions = predictions[0]

                boxes = predictions["boxes"].cpu().numpy().astype(np.int32)
                scores = predictions["scores"].cpu().numpy()
                boxes, scores = non_maximum_suppression(boxes, scores)
                character_names = [characters[ch] for ch in predictions["labels"].cpu().numpy()]
                draw_rectangles_on_image_and_save(image, [(box, (0, 255, 0)) for box in boxes], f"fasterRCNNpredictions/{image_name}", character_names)

                for box in boxes:
                    detections.append(box[0])
                    detections.append(box[1])
                    detections.append(box[2])
                    detections.append(box[3])
                for s in scores:
                    final_scores.append(s)
                for _ in range(len(boxes)):
                    file_names.append(image_name)


                for box, score, character in zip(boxes, scores, character_names):
                    if character == "barney":
                        barney_detections.append(box[0])
                        barney_detections.append(box[1])
                        barney_detections.append(box[2])
                        barney_detections.append(box[3])
                        barney_detections_file_names.append(image_name)
                        barney_detections_scores.append(score)
                    elif character == "betty":
                        betty_detections.append(box[0])
                        betty_detections.append(box[1])
                        betty_detections.append(box[2])
                        betty_detections.append(box[3])
                        betty_detections_file_names.append(image_name)
                        betty_detections_scores.append(score)
                    elif character == "fred":
                        fred_detections.append(box[0])
                        fred_detections.append(box[1])
                        fred_detections.append(box[2])
                        fred_detections.append(box[3])
                        fred_detections_file_names.append(image_name)
                        fred_detections_scores.append(score)
                    elif character == "wilma":
                        wilma_detections.append(box[0])
                        wilma_detections.append(box[1])
                        wilma_detections.append(box[2])
                        wilma_detections.append(box[3])
                        wilma_detections_file_names.append(image_name)
                        wilma_detections_scores.append(score)

        detections = np.array(detections)
        detections = detections.reshape((detections.shape[0] // 4, 4))
        final_scores = np.array(final_scores)
        file_names = np.array(file_names)

        barney_detections = np.array(barney_detections)
        barney_detections = barney_detections.reshape((barney_detections.shape[0] // 4, 4))
        barney_detections_scores = np.array(barney_detections_scores)
        barney_detections_file_names = np.array(barney_detections_file_names)

        betty_detections = np.array(betty_detections)
        betty_detections = betty_detections.reshape((betty_detections.shape[0] // 4, 4))
        betty_detections_scores = np.array(betty_detections_scores)
        betty_detections_file_names = np.array(betty_detections_file_names)

        fred_detections = np.array(fred_detections)
        fred_detections = fred_detections.reshape((fred_detections.shape[0] // 4, 4))
        fred_detections_scores = np.array(fred_detections_scores)
        fred_detections_file_names = np.array(fred_detections_file_names)

        wilma_detections = np.array(wilma_detections)
        wilma_detections = wilma_detections.reshape((wilma_detections.shape[0] // 4, 4))
        wilma_detections_scores = np.array(wilma_detections_scores)
        wilma_detections_file_names = np.array(wilma_detections_file_names)


        eval_detections(detections, final_scores, file_names, plot_name="fasterrcnn")

        eval_detections(np.array(barney_detections), np.array([1] * len(barney_detections)),
                        np.array(barney_detections_file_names), "validare/task2_barney_gt_validare.txt", "RCNNbarney")
        eval_detections(np.array(betty_detections), np.array([1] * len(betty_detections)),
                        np.array(betty_detections_file_names), "validare/task2_betty_gt_validare.txt", "RCNNbetty")
        eval_detections(np.array(fred_detections), np.array([1] * len(fred_detections)),
                        np.array(fred_detections_file_names), "validare/task2_fred_gt_validare.txt", "RCNNfred")
        eval_detections(np.array(wilma_detections), np.array([1] * len(wilma_detections)),
                        np.array(wilma_detections_file_names), "validare/task2_wilma_gt_validare.txt", "RCNNwilma")

    def predict(self):
        pass

    def save(self):
        torch.save(self.model.state_dict(), "models/fasterRCNN.pth")


model = MyFasterRCNN()
# model.train()
model.test()