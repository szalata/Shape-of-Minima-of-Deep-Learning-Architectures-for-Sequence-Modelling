# -*- coding: utf-8 -*-

import torch
import numpy as np
from pyhessian import hessian
from SeqDataset import SeqDataset
from torch.utils.data import DataLoader
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--top_eig', type=int, default=10, help='number of highest absolute value eigenvalues to consider')
parser.add_argument('--model_dir', type=str, help='path to model',
                        default="trained_models/model.pt")
parser.add_argument('--output_dir', type=str, help='name of the file to write', default="data")
args = parser.parse_args()

model = torch.load(args.model_dir)
model.eval()
criterion = torch.nn.BCELoss()

inputs = np.load("data/sequence_classification/X_train.npy")
dataset_train = SeqDataset("data/sequence_classification", "train")

dataloader_train = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=inputs.shape[0])
for inputs, targets in dataloader_train:
    hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)
    
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=args.top_eig, maxIter=500)

# analysis
max_eig = top_eigenvalues[0]
mean_eig = np.mean(np.abs(top_eigenvalues))
trace = np.mean(hessian_comp.trace(maxIter=500))

print("Maximum eigenvalue is ", max_eig)
print("Mean of maximum ", args.top_eig, " eigenvalues is ", mean_eig)
print("Trace of Hessian is ", trace)

# saving results
np.save(args.output_dir + "/max_ieg.npy", max_eig)
np.save(args.output_dir + "/top_eig.npy", mean_eig)
np.save(args.output_dir + "/trace.npy", trace)

