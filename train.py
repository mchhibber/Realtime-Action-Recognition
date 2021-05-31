import torch.nn as nn
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

from utils import *
import datasets.transforms as T
from datasets.dataset import *


def get_train_loader(video_path, annotation_path, dataset_name):
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {
        'num_workers': 2}

    train_transforms = torchvision.transforms.Compose([T.ToFloatTensorInZeroOne(),
                                  T.Resize((128, 128)),
                                  T.RandomHorizontalFlip(),
                                  T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                                  T.CenterCrop((112, 112))
                                  ])

    train_dataset = get_dataset(video_path, annotation_path, dataset_name, train_transforms, True)
    val_split = 0.05
    total_train_samples = len(train_dataset)
    total_val_samples = round(total_train_samples * val_split)
    train, val = random_split(train_dataset, [total_train_samples - total_val_samples,
                                              total_val_samples])

    train_loader = DataLoader(train, batch_size=64, shuffle=True, **kwargs)
    val_loader = DataLoader(val, batch_size=64, shuffle=True, **kwargs)
    return train_loader, val_loader


def train_epoch(epoch, data_loader, model, optimizer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end_time = time.time()

    for i, data in enumerate(data_loader):
        inputs, targets = data[0], data[-1]
        data_time.update(time.time() - end_time)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model)
        model.to(device)
        targets.to(device)
        inputs.to(device)
        inputs = Variable(inputs)
        targets = Variable(targets)

        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        losses.update(loss.data, inputs.size(0))

        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
                lr=optimizer.param_groups[0]['lr']))


def val_epoch(epoch, data_loader, model):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, targets = data[0], data[-1]
            data_time.update(time.time() - end_time)

            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()

            inputs = Variable(inputs)
            targets = Variable(targets)
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            losses.update(loss.data, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))
        return losses.avg.item(), top1.avg.item()


if __name__ == "__main__":
    train_transforms = torchvision.transforms.Compose([T.ToFloatTensorInZeroOne(),
                                                       T.Resize((128, 128)),
                                                       T.RandomHorizontalFlip(),
                                                       T.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
                                                       T.CenterCrop((112, 112))
                                                       ])