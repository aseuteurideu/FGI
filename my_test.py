import argparse
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str,
                    help='Folder location of the test results')
parser.add_argument('--dataset', type=str, default='dfdc', choices=['dfdc', 'fakeavceleb'], help='Dataset selection')
args = parser.parse_args()

orig_stdout = sys.stdout
f = open(os.path.join(args.folder, 'output.txt'), 'a')
sys.stdout = f

print('')
print('')
print(args)

print('test on dataset: ', args.dataset)

dataset = '' if args.dataset == 'dfdc' else args.dataset

# file_pred.pkl contains a dictionary with key as video name and value as the sum of
# preds of all chunks of that video
with open(os.path.join(args.folder, dataset + 'file_pred.pkl'), 'rb') as handle:
    test_pred = pickle.load(handle)

# file_target.pkl contains a dictionary with key as video name and value as the true target (real/fake)
# for that video
with open(os.path.join(args.folder, dataset + 'file_target.pkl'), 'rb') as handle:
    test_target = pickle.load(handle)

# file_number_of_chunks.pkl contains a dictionary with key as video name and value as the number of chunks
# in that video
with open(os.path.join(args.folder, dataset + 'file_number_of_chunks.pkl'), 'rb') as handle:
    test_number_of_chunks = pickle.load(handle)

pred_fake = []
pred_real = []

for video, score in test_pred.items():
    tar = test_target[video]
    score = test_pred[video]
    num_chunks = test_number_of_chunks[video]
    mean_pred = (score) / num_chunks

    if tar == 1:
        pred_fake.append(mean_pred)
    else:
        pred_real.append(mean_pred)

pred_fake = np.array(pred_fake)
pred_real = np.array(pred_real)
gt_fake = np.ones_like(pred_fake)
gt_real = np.zeros_like(pred_real)
pred_all = np.concatenate([pred_fake, pred_real])
gt_all = np.concatenate([gt_fake, gt_real])
auc = roc_auc_score(gt_all, pred_all)
print('auc = ', auc)

sys.stdout = orig_stdout
f.close()