## code source - https://github.com/TengdaHan/DPC/blob/master/utils/utils.py
from datetime import datetime
from model import *
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from collections import deque
from torchvision import transforms

def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value + 0.00000001)

def save_checkpoint(state, is_best=0, gap=1, filename='models/checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename),
                                   'epoch%s.pth.tar' % str(state['epoch'] - gap))
    if (state['epoch'] - gap) == 50:  # keep the 50th epoch result. change the path. Move it to halfepochs_results
        os.makedirs(os.path.join(os.path.dirname(filename), 'halfepochs_results'), exist_ok=True)
        os.rename(last_epoch_path,
                  os.path.join(os.path.dirname(filename), 'halfepochs_results', 'epoch%s.pth.tar' % str(state['epoch'] - gap)))
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        for i in past_best:
            os.rename(i,
                      os.path.join(os.path.dirname(filename), 'halfepochs_results', os.path.basename(i)))
    if not keep_all:
        try:
            os.remove(last_epoch_path)
        except:
            pass
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        for i in past_best:
            try:
                os.remove(i)
            except:
                pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()


def calc_accuracy(vid_out, aud_out, target, threshold):
    batch_size = target.size(0)
    pred = 0
    correct = 0
    for batch in range(batch_size):
        dist = torch.dist(vid_out[batch, :].view(-1), aud_out[batch, :].view(-1), 2)
        tar = target[batch, :].view(-1).item()
        if dist < threshold:
            pred = 1
        else:
            pred = 0
        if pred == tar:
            correct += 1
    return correct * (1 / batch_size)



def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {}  # save all data values here
        self.save_dict = {}  # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count

