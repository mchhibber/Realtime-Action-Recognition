import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import ActionRecognitionModel
from torch.utils.data import DataLoader, random_split
import datasets.transforms as T
from datasets.dataset import *
from test import *
from train import *
from utils import *

if __name__ == "__main__":
    dataset = 'HMDB51'
    resume = False
    train = True

    video_path, annotation_path, num_classes = get_dataset_info(dataset)
    model = ActionRecognitionModel(num_classes, sample_size=112, width_mult=1.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        dampening=0.9,
        weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=7)
    if resume:
        checkpoint = torch.load('HAR.pth', map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_prec1 = checkpoint['best_prec1']

    if train:
        train_loader, val_loader = get_train_loader(video_path, annotation_path, dataset)
        print("Launching Action Recognition Model training")
        for i in range(30):
            train_epoch(i, train_loader, model, optimizer)
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, False, 'HAR')
            _, prec1 = val_epoch(val_loader, model)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, is_best, 'HAR')
    else:
        test_loader = get_test_loader(video_path, annotation_path, dataset)
        test(test_loader, model)
