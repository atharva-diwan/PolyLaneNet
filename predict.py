import cv2
import os
import sys
import random
import logging
import argparse
import subprocess
from time import time
import numpy as np
from lib.config import Config
from utils.evaluator import Evaluator
import numpy as np
import torch
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class Prediction_class(Dataset):
    def __init__(self,
                 img_dir,
                 normalize=True,
                 img_size=(720, 1080)):

        self.img_h, self.img_w = img_size
        self.normalize = normalize
        self.to_tensor = ToTensor()
        self.img_dir = img_dir

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[0])
        img = cv2.imread(img_path)
        # print("*"*50)
        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, idx)

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def draw_annotation(self, idx, pred=None, img=None, cls_pred=None):

        img, _ = self.__getitem__(idx)
        # Tensor to opencv image
        img = img.permute(1, 2, 0).numpy()
        # Unnormalize

        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = (img * 255).astype(np.uint8)

        img_h, img_w, _ = img.shape

        print(pred)
        # pred = pred.reshape(5,7)
        # Draw predictions
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        # matches, accs, _ = self.get_metrics(pred, idx)
        overlay = img.copy()
        for i, lane in enumerate(pred):

            # print(lane.shape)
            color = PRED_HIT_COLOR

            pred_conf = lane[0]
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions

            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

            # draw class icon
            if cls_pred is not None and len(points) > 0:
                class_icon = self.dataset.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

            # draw lane ID
            if len(points) > 0:
                cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)

            # draw lane accuracy
            if len(points) > 0:
                cv2.putText(img,
                            '{:.2f}'.format(pred_conf),
                            tuple(points[len(points) // 2] - 30),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=.75,
                            color=color)
        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img

cfg = Config("experiments/tusimple_efficientnetb1/config.yaml")
# Set up seeds
torch.manual_seed(cfg['seed'])
np.random.seed(cfg['seed'])
random.seed(cfg['seed'])



# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = cfg["epochs"]
batch_size = cfg["batch_size"]

model = cfg.get_model().to(device)

img_dir = 'data_lane-regression/datasets/prediction'
predict_set = Prediction_class(img_dir, normalize=True, img_size=(720, 1080))

test_loader = torch.utils.data.DataLoader(dataset=predict_set,
                                              batch_size=1,
                                              shuffle=False)
model.load_state_dict(torch.load('experiments/tusimple_efficientnetb1/models/model_2695.pt',
                                 map_location=torch.device('cpu'))['model'])

test_parameters = cfg.get_test_parameters()

with torch.no_grad():
    for idx, (images, img_idxs) in enumerate(test_loader):
        images = images.to(device)

        outputs = model(images)

        labels = np.zeros((1, 1), dtype=int)
        outputs = model.decode(outputs, labels, **test_parameters)

        output, extra_outputs = outputs
        # print(output.shape)
        # print(images.shape)
        preds = test_loader.dataset.draw_annotation(idx, pred=output[0].cpu().numpy(),
                                                    cls_pred=extra_outputs[0].cpu().numpy() if extra_outputs is not None else None)



cv2.imshow('pred', preds)
cv2.waitKey(0)
cv2.destroyAllWindows()