import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from model import CrystalGraphConvNet  # 用的是你已修改支持 freeze_conv 的 model.py

# ========== 参数定义 ==========
parser = argparse.ArgumentParser()
parser.add_argument('data_options', metavar='OPTIONS', nargs='+', help='path to root dir followed by options')
parser.add_argument('--task', choices=['regression', 'classification'], default='regression')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight-decay', default=0, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int)
parser.add_argument('--optim', default='Adam', choices=['Adam', 'SGD'])
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--freeze-conv', action='store_true', help='Freeze convolution layers')
parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model')

# 模型结构参数
parser.add_argument('--atom-fea-len', default=64, type=int)
parser.add_argument('--h-fea-len', default=128, type=int)
parser.add_argument('--n-conv', default=3, type=int)
parser.add_argument('--n-h', default=1, type=int)

# 数据划分参数
parser.add_argument('--train-size', default=None, type=int)
parser.add_argument('--val-size', default=None, type=int)
parser.add_argument('--test-size', default=None, type=int)

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# ========== 数据加载 ==========
dataset = CIFData(*args.data_options)
collate_fn = collate_pool
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset=dataset,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    train_size=args.train_size,
    val_size=args.val_size,
    test_size=args.test_size,
    return_test=True,
    pin_memory=args.cuda
)

# ========== 标准化器 ==========
class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed):
        return normed * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state):
        self.mean = state['mean']
        self.std = state['std']

if args.task == 'classification':
    normalizer = Normalizer(torch.zeros(2))
    normalizer.load_state_dict({'mean': 0., 'std': 1.})
else:
    sample_data = [dataset[i] for i in sample(range(len(dataset)), min(len(dataset), 500))]
    _, sample_target, _ = collate_pool(sample_data)
    normalizer = Normalizer(sample_target)

# ========== 模型构建 ==========
sample_data = dataset[0]
orig_atom_fea_len = sample_data[0].shape[-1]
nbr_fea_len = sample_data[1].shape[-1]

model = CrystalGraphConvNet(
    orig_atom_fea_len=orig_atom_fea_len,
    nbr_fea_len=nbr_fea_len,
    atom_fea_len=args.atom_fea_len,
    n_conv=args.n_conv,
    h_fea_len=args.h_fea_len,
    n_h=args.n_h,
    classification=args.task == 'classification',
    freeze_conv=args.freeze_conv
)

if args.cuda:
    model.cuda()

# ========== 预训练模型加载（迁移学习）==========
if args.pretrained and os.path.isfile(args.pretrained):
    print(f"=> Loading pretrained model from {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# ========== 优化器和损失函数 ==========
criterion = nn.MSELoss() if args.task == 'regression' else nn.NLLLoss()
opt_params = model.get_trainable_parameters()  # 迁移学习只优化未冻结部分
optimizer = (optim.Adam(opt_params, lr=args.lr, weight_decay=args.weight_decay)
             if args.optim == 'Adam'
             else optim.SGD(opt_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay))
scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

# ========== 训练/验证函数 ==========
def mae(pred, target): return torch.mean(torch.abs(pred - target))

def run_epoch(loader, model, training=True):
    model.train() if training else model.eval()
    total_loss = 0
    total_mae = 0
    total_count = 0
    for atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, target, _ in loader:
        if args.cuda:
            atom_fea, nbr_fea, nbr_fea_idx = atom_fea.cuda(), nbr_fea.cuda(), nbr_fea_idx.cuda()
            target = target.cuda()
            crystal_atom_idx = [i.cuda() for i in crystal_atom_idx]

        out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(out, normalizer.norm(target))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * target.size(0)
        total_mae += mae(normalizer.denorm(out), target).item() * target.size(0)
        total_count += target.size(0)

    return total_loss / total_count, total_mae / total_count

# ========== 主训练循环 ==========
best_mae = float('inf')
for epoch in range(args.epochs):
    train_loss, train_mae = run_epoch(train_loader, model, training=True)
    val_loss, val_mae = run_epoch(val_loader, model, training=False)

    print(f"Epoch {epoch+1:03d}: Train MAE={train_mae:.4f}, Val MAE={val_mae:.4f}")

    if val_mae < best_mae:
        best_mae = val_mae
        torch.save({'model_state_dict': model.state_dict()}, 'model_best.pth.tar')
        print(">> Model saved.")

    scheduler.step()

# ========== 测试最优模型 ==========
print("==> Testing best model...")
checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
_, test_mae = run_epoch(test_loader, model, training=False)
print(f"Final Test MAE = {test_mae:.4f}")

