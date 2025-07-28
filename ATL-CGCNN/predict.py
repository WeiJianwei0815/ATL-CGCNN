# -*- coding: utf-8 -*-

import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <data_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load dataset
    dataset = CIFData(data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_pool)

    # 2. Initialize model
    sample_input = dataset[0][0]
    orig_atom_fea_len = sample_input[0].shape[1]
    nbr_fea_len = sample_input[1].shape[2]

    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 3. Run prediction
    predictions = []
    targets = []
    cif_ids = []

    with torch.no_grad():
        for input, target, batch_cif_ids in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [c.to(device) for c in crystal_atom_idx]

            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            predictions.append(output.item())
            targets.append(target.item())
            cif_ids.append(batch_cif_ids[0])

    # 4. Save prediction results
    df = pd.DataFrame({
        'cif_id': cif_ids,
        'true_value': targets,
        'predicted_value': predictions
    })
    df.to_csv('prediction_results.csv', index=False)

    # 5. Print metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    print("Prediction complete.")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("Results saved to prediction_results.csv")

    # 6. Plot true vs predicted
    plt.figure()
    plt.scatter(targets, predictions, alpha=0.7)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("True vs Predicted")
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("true_vs_predicted.png")
    print("Scatter plot saved as true_vs_predicted.png")

if __name__ == '__main__':
    main()

