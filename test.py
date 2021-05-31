import torch.nn as nn
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets.dataset import *

from utils import *


def get_test_loader(video_path, annotation_path, dataset_name):
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {
        'num_workers': 2}


    test_dataset = get_dataset(video_path, annotation_path, dataset_name, train=False)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)

    return test_loader


def test(data_loader, model):
    print('test')

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

            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))

        return top1.avg.item()

