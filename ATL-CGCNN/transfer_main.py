import argparse
import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser(description='Transfer Learning for CGCNN')
parser.add_argument('data_path', metavar='DP', help='Path to target dataset')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='Initial learning rate')
parser.add_argument('--lr-ratio', default=0.1, type=float, help='Learning rate scaling for earlier layers')
parser.add_argument('--train-ratio', default=0.7, type=float, help='Training data ratio')
parser.add_argument('--val-ratio', default=0.15, type=float, help='Validation data ratio')
parser.add_argument('--test-ratio', default=0.15, type=float, help='Test data ratio')
parser.add_argument('--pretrained-path', default='pretrained_cgcnn.pth', help='Path to pretrained model')
parser.add_argument('--freeze-until-layer', default=3, type=int, help='Freeze up to conv layer n (0-3)')
parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
parser.add_argument('--task', default='regression', help='regression or classification')
parser.add_argument('--hidden-size', default=64, type=int, help='Hidden feature size')
parser.add_argument('--n-conv', default=3, type=int, help='Number of convolutional layers')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate') 
args = parser.parse_args()

# ---------------------------
# Set Random Seed and Device
# ---------------------------
random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

# ---------------------------
# Load Target Dataset
# ---------------------------
dataset = CIFData(args.data_path)
collate_fn = collate_pool
train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    train_ratio=args.train_ratio,
    val_ratio=args.val_ratio,
    test_ratio=args.test_ratio,
    return_test=True,
    num_workers=0,
    pin_memory=args.cuda,
    train_size=None,
    val_size=None,
    test_size=None
)

# ---------------------------
# Build Model and Load Pretrained
# ---------------------------
sample_data = dataset[0][0]
orig_atom_fea_len = sample_data[0].shape[1]
nbr_fea_len = sample_data[1].shape[2]

model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len).to(device)
pretrained_data = torch.load(args.pretrained_path, map_location=device)
if 'state_dict' in pretrained_data:
    model.load_state_dict(pretrained_data['state_dict'])
else:
    model.load_state_dict(pretrained_data)

# ---------------------------
# Freeze Layers & Adjust Learning Rate
# ---------------------------
print("[Freezing Layer Strategy and Learning Rate Grouping]")
param_groups = [
    {'params': [], 'lr': args.lr * args.lr_ratio},
    {'params': [], 'lr': args.lr}
]

high_lr_ids = set()
for i, conv in enumerate(model.convs):
    for param in conv.parameters():
        if i < args.freeze_until_layer:
            param.requires_grad = False
            print(f"conv{i+1}: Frozen")
        else:
            param.requires_grad = True
            param_groups[1]['params'].append(param)
            high_lr_ids.add(id(param))
            print(f"conv{i+1}: Trainable (lr={args.lr})")

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.startswith('fc_full') or name.startswith('fc_out'):
        param_groups[1]['params'].append(param)
        high_lr_ids.add(id(param))
        print(f"{name}: Trainable (lr={args.lr})")
    elif 'convs' in name and id(param) not in high_lr_ids:
        param_groups[0]['params'].append(param)
        print(f"{name}: Assigned to low-lr group (lr={args.lr * args.lr_ratio})")

# ---------------------------
# Optimizer and Loss
# ---------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(param_groups)

# ---------------------------
# Training Loop
# ---------------------------
def train(epoch):
    model.train()
    loss_all = 0
    for i, (input, target, _) in enumerate(train_loader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
        input = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(*input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return loss_all / len(train_loader)

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input, target, _ in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
            input = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            target = target.to(device)
            output = model(*input)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)

# ---------------------------
# Run Training with Logging
# ---------------------------
train_losses = []
val_losses = []
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    val_loss = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'args': {
                'task': 'regression',
                'atom_fea_len': orig_atom_fea_len,
                'nbr_fea_len': nbr_fea_len,
                'hidden_size': args.hidden_size,
                'n_conv': args.n_conv,
                'dropout': args.dropout
            },
        }, 'atl_cgcnn_best.pth')
        print("[Info] Best model saved.")

# ---------------------------
# Save Logs and Plot
# ---------------------------
log_df = pd.DataFrame({
    'Epoch': list(range(1, args.epochs + 1)),
    'Train Loss': train_losses,
    'Val Loss': val_losses
})
log_df.to_excel("training_log.xlsx", index=False)

plt.figure()
plt.plot(log_df['Epoch'], log_df['Train Loss'], label='Train Loss')
plt.plot(log_df['Epoch'], log_df['Val Loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

# ---------------------------
# Final Evaluation
# ---------------------------
checkpoint = torch.load('atl_cgcnn_best.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

true_vals = []
pred_vals = []

with torch.no_grad():
    for input, target, _ in test_loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
        input = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        target = target.to(device)
        output = model(*input)
        true_vals.append(target.cpu().numpy())
        pred_vals.append(output.cpu().numpy())

true_vals = np.concatenate(true_vals).flatten()
pred_vals = np.concatenate(pred_vals).flatten()

mse = np.mean((true_vals - pred_vals)**2)
mae = mean_absolute_error(true_vals, pred_vals)
print(f"Final Test MSE: {mse:.4f}, MAE: {mae:.4f}")

np.savetxt("prediction_results.csv", np.vstack((true_vals, pred_vals)).T, delimiter=",", header="True,Pred", comments='')
