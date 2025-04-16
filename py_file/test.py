use_gpu = True

import numpy as np
import pandas as pd
from tqdm import tqdm
from gradcam import runGradCam
import heapq

import os
import sys
sys.path.insert(0, '..')

if use_gpu:
    from utils.gpu_tools import *
    # selected_gpus = select_gpu(query_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([ str(obj) for obj in select_gpu(query_gpu())])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")


# os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

torch.manual_seed(42)

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}  

year_list = np.arange(2001,2020,1)
images = []
label_df = []
for year in year_list:
    images.append(np.memmap(os.path.join("../monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8, mode='r').reshape(
                        (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    label_df.append(pd.read_feather(os.path.join("../monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))
    
images = np.concatenate(images)
label_df = pd.concat(label_df)

print(images.shape)
print(label_df.shape)

class MyDataset(Dataset):
    
    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)
  
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]
    
dataset = MyDataset(images, (label_df.Ret_20d > 0).values)

test_dataloader = DataLoader(dataset, batch_size=2048, shuffle=False)

net_path = '../pt/20250416_17:06:38/baseline_epoch_25_train_0.670839_val_0.701352.pt' # need to rewrite

from models import baseline
device = 'cuda' if use_gpu else 'cpu'
print(f"Using device: {device}")

net = torch.load(net_path, weights_only=False)
# net = nn.DataParallel(net)
net = net.to(device)

def eval_loop(dataloader, net, loss_fn):
    
    running_loss = 0.0
    total_loss = 0.0
    current = 0
    net.eval()
    target = []
    predict = []
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)
                y_pred = net(X)
                target.append(y.detach())
                predict.append(y_pred.detach())
                loss = loss_fn(y_pred, y.long())
                running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
                current += len(X)
                t.set_postfix({'running_loss':running_loss})
            
    return total_loss, torch.cat(predict), torch.cat(target)

print('test start')
loss_fn = nn.CrossEntropyLoss()
test_loss, y_pred, y_target = eval_loop(test_dataloader, net, loss_fn)

predict_logit = (torch.nn.Softmax(dim=1)(y_pred)[:,1]).cpu().numpy()    

from matplotlib import pyplot as plt

threshold = 0.

label_df['ret'] = (predict_logit>threshold) * label_df.Ret_20d
label_filtered = label_df[predict_logit>threshold]
ret_baseline = label_filtered .groupby(['Date'])['Ret_20d'].mean()

threshold = 0.58

label_df['ret'] = (predict_logit>threshold) * label_df.Ret_20d
label_filtered = label_df[predict_logit>threshold]
ret_cnn = label_filtered .groupby(['Date'])['Ret_20d'].mean()

plt.scatter(label_filtered.groupby(['Date'])['ret'].count().index, label_filtered.groupby(['Date'])['ret'].count(),marker='+')
plt.savefig('../pic/scatter_plot.png', dpi=300, facecolor='w')  # Save scatter plot
plt.close()  # Close the figure to free memory

log_ret_baseline = np.log10((ret_baseline+1).cumprod().fillna(method='ffill'))
log_ret_cnn = np.log10((ret_cnn+1).cumprod().fillna(method='ffill'))
fig = plt.figure()
plt.plot(log_ret_baseline, label='baseline')
plt.plot(log_ret_cnn, label='CNN')
plt.plot(log_ret_cnn - log_ret_baseline, alpha=0.6, lw=2, label='exceed_ret')
plt.legend()
plt.savefig('../pic/performance1.png', dpi = 300, facecolor='w')
plt.close() 
# plt.show()
# fig.savefig('../pic/performance1.png',dpi=300)

plt.plot((ret_cnn+1).cumprod().fillna(method='ffill'), label='CNN_accumulate_ret')
plt.legend()
plt.savefig('../pic/performance2.png', dpi = 300, facecolor='w')
plt.close() 
# plt.savefig('../pic/performance2.png',dpi=300)

label_df['weighted_ret'] = 1 * label_df.Ret_20d * label_df['EWMA_vol']
label_df['weight'] = 1 * label_df['EWMA_vol']
ret_baseline = label_df.groupby(['Date'])['weighted_ret'].sum()/(label_df.groupby(['Date'])['weight'].sum())

threshold = 0.58

label_df['weighted_ret'] = (predict_logit>threshold) * label_df.Ret_20d * label_df['EWMA_vol']
label_df['weight'] = (predict_logit>threshold) * label_df['EWMA_vol']
ret_cnn = label_df.groupby(['Date'])['weighted_ret'].sum()/(label_df.groupby(['Date'])['weight'].sum())

log_ret_baseline = np.log10((ret_baseline+1).cumprod().fillna(method='ffill'))
log_ret_cnn = np.log10((ret_cnn+1).cumprod().fillna(method='ffill'))
plt.plot(log_ret_baseline)
plt.plot(log_ret_cnn)
plt.plot(log_ret_cnn - log_ret_baseline, alpha=0.6, lw=2)
plt.legend()
plt.savefig('../pic/EWMA_Vol.png', dpi = 300, facecolor='w')
plt.close()

results_path = '../pic'
os.makedirs(os.path.join(results_path, 'gradcam', 'down'), exist_ok=True)
os.makedirs(os.path.join(results_path, 'gradcam', 'up'), exist_ok=True)

def generate_top5_samples(y_pred, y_target, images, criterion=None):
    """
    Generate top 5 samples for Grad-CAM analysis.
    For classification tasks, selects the top 5 samples with the highest probabilities for each class.
    For regression tasks, selects the top 5 samples with the lowest absolute relative errors.
    """
    high_scores_0 = []  # For class 0 (down)
    high_scores_1 = []  # For class 1 (up)

    if criterion is None or not isinstance(criterion, nn.MSELoss):
        # Classification task
        scores = torch.softmax(y_pred, dim=1)  # Compute probabilities
        score, pred = torch.max(scores, dim=1)  # Get max probability and predicted class

        correct = pred == y_target  # Correctly classified samples
        correct_scores = scores[correct]
        correct_classes = y_target[correct]
        correct_images = images[correct.cpu()]   # Filter corresponding images
        correct_images = correct_images.to(device) 
        # correct_images = images[correct]  # Filter corresponding images

        for idx, (score, clazz, img) in enumerate(zip(correct_scores, correct_classes, correct_images)):
            if clazz == 0:  # Class 0 (down)
                if len(high_scores_0) < 5:
                    heapq.heappush(high_scores_0, (score[0].item(), img))
                else:
                    heapq.heappushpop(high_scores_0, (score[0].item(), img))
            elif clazz == 1:  # Class 1 (up)
                if len(high_scores_1) < 5:
                    heapq.heappush(high_scores_1, (score[1].item(), img))
                else:
                    heapq.heappushpop(high_scores_1, (score[1].item(), img))
    else:
        # Regression task
        scores = -torch.abs((y_pred - y_target) / y_target)  # Relative error
        for idx, (score, true_value, img) in enumerate(zip(scores, y_target, images)):
            if true_value < 0:  # Downward case
                if len(high_scores_0) < 5:
                    heapq.heappush(high_scores_0, (score.item(), img))
                else:
                    heapq.heappushpop(high_scores_0, (score.item(), img))
            elif true_value > 0:  # Upward case
                if len(high_scores_1) < 5:
                    heapq.heappush(high_scores_1, (score.item(), img))
                else:
                    heapq.heappushpop(high_scores_1, (score.item(), img))

    # Sort the top 5 samples by score in descending order
    high_scores_0 = sorted(high_scores_0, reverse=True, key=lambda x: x[0])
    high_scores_1 = sorted(high_scores_1, reverse=True, key=lambda x: x[0])

    return high_scores_0, high_scores_1

# images = torch.tensor(images, dtype=torch.float32)
# images = images.to(device)
top5_down, top5_up = generate_top5_samples(y_pred, y_target, dataset.img, loss_fn)

# Define output dimension
outdim = 2  # Assume output dimension is 2, adjust based on the actual model

# Grad-CAM result generation
savepath = os.path.join(results_path, 'gradcam', 'down')
runGradCam(net.module, top5_down, 0, savepath)  # Generate Grad-CAM for the 'down' class

savepath = os.path.join(results_path, 'gradcam', 'up')
if outdim == 2:
    runGradCam(net.module, top5_up, 1, savepath)  # Generate Grad-CAM for the 'up' class
elif outdim == 1:
    runGradCam(net.module, top5_up, 0, savepath)  # If single output dimension
