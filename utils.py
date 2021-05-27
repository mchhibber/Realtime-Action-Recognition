import torch
import shutil


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_acc(self, val, n=1):
        self.val = val / n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, path):
    torch.save(state, '%s.pth' % path)
    if is_best:
        shutil.copyfile('%s.pth' % path, '%s_best.pth' % path)
