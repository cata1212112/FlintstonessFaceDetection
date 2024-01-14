import joblib
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN, backbone_utils
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import *
from PIL import Image
import os
import numpy as np
from torchvision.transforms import v2 as T
from tqdm import tqdm
from utility import non_maximum_suppression, draw_rectangles_on_image_and_save
from Save import SaveSolution

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

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        image_id = torch.tensor([item])
        target["image_id"] = image_id

        image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_names)


class MyFasterRCNN:
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 5
    CLASSES = ['barney', 'betty', 'fred', 'wilma', 'unknown']
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self):
        custom_backbone = backbone_utils.resnet_fpn_backbone(backbone_name='resnet50', weights=None, trainable_layers=5)
        self.model = FasterRCNN(backbone=custom_backbone, num_classes=MyFasterRCNN.NUM_CLASSES + 1, min_size=240,
                                max_size=240)
        total_params = sum(param.numel() for param in self.model.parameters())
        print(f"{total_params:,} trainable parameters.")
        self.model.to(MyFasterRCNN.DEVICE)
        self.writer = SummaryWriter(log_dir='runs', comment='FasterRCNN')

    def load_data(self, load=False):
        if load:
            training_data = joblib.load("training_data.joblib")
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
                training_data[image_name] = boxes[image_name]

            joblib.dump(training_data, "training_data.joblib")

        print(f"Training images: {len(training_data)}")

        return FlintstonesDataset(training_data, train_transforms)

    def train(self):
        train_dataset = self.load_data(load=False)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=MyFasterRCNN.BATCH_SIZE, shuffle=True,
                                                        collate_fn=lambda x: list(zip(*x)))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=MyFasterRCNN.LEARNING_RATE)

        self.model.train()
        for epoch in range(MyFasterRCNN.EPOCHS):
            print(f"Epoch {epoch + 1}/{MyFasterRCNN.EPOCHS}")
            train_losses = []
            for images, targets in train_data_loader:
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

            self.save()

    def test(self):
        self.model.load_state_dict(torch.load("models/fasterRCNN.pth"))
        self.model.eval()

        detections = []
        final_scores = []
        file_names = []

        character_detections = {'barney': [], 'betty': [], 'fred': [], 'wilma': []}
        character_detections_file_names = {'barney': [], 'betty': [], 'fred': [], 'wilma': []}
        chatacter_detection_scores = {'barney': [], 'betty': [], 'fred': [], 'wilma': []}

        for image_name in tqdm(os.listdir("validare/validare"), desc="Detecting with FasterRCNN"):
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
                if DRAW:
                    draw_rectangles_on_image_and_save(image, [(box, (0, 255, 0)) for box in boxes],
                                                      f"fasterRCNNpredictions/{image_name}")

                for box in boxes:
                    detections.append(box)
                for s in scores:
                    final_scores.append(s)
                for _ in range(len(boxes)):
                    file_names.append(image_name)

                for box, score, character in zip(boxes, scores, character_names):
                    if character == "unknown":
                        continue

                    character_detections[character].append(box)

                    character_detections_file_names[character].append(image_name)
                    chatacter_detection_scores[character].append(score)

        detections = np.array(detections)
        final_scores = np.array(final_scores)
        file_names = np.array(file_names)

        SaveSolution("fisiere_solutie/bonus", "all_faces", detections, final_scores, file_names).save()

        for character in characters:
            if character == "unknown":
                continue
            SaveSolution("fisiere_solutie/bonus", character, character_detections[character],
                         chatacter_detection_scores[character],
                         character_detections_file_names[character]).save()

    def save(self):
        torch.save(self.model.state_dict(), "models/fasterRCNN.pth")