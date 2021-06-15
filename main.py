# coding: utf-8
import argparse
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# imports for hessian
from pyhessian import hessian
from PyHessian.density_plot import get_esd_plot

# imports for visualization
import copy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = [18, 12]
import loss_landscapes
import loss_landscapes.metrics

import model
from SeqDataset import SeqDataset

parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=1,
                    help='size of input embeddings')
parser.add_argument('--nhid', type=int, default=8,
                    help='Hidden embedding size')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--nout', type=int, default=1,
                    help='number of outputs')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=160, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='output',
                    help='Output path')
parser.add_argument('--model_path', type=str, default=None,
                    help='path to load the model')
parser.add_argument('--data_dir', type=str, default="data/sequence_classification/varied_length",
                    help='directory with the dataset')
parser.add_argument('--nhead', type=int, default=1,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()
Path(args.save).mkdir(parents=True, exist_ok=True)

args.task = args.data_dir.split("/")[1]

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################
dataset_train = SeqDataset(args.data_dir, "train")
dataset_val = SeqDataset(args.data_dir, "val")
dataset_test = SeqDataset(args.data_dir, "test")
dataloader_train = DataLoader(dataset_train,
                              shuffle=True,
                              collate_fn=dataset_train.collate_fn,
                              batch_size=args.batch_size)
dataloader_val = DataLoader(dataset_val,
                            shuffle=False,
                            collate_fn=dataset_val.collate_fn,
                            batch_size=args.batch_size)
dataloader_test = DataLoader(dataset_test,
                             shuffle=False,
                             collate_fn=dataset_test.collate_fn,
                             batch_size=args.batch_size)

if args.model == 'Transformer':
    model = model.TransformerModel(args.emsize, args.nhead, args.nhid, args.nlayers, args.task,
                                   args.dropout).to(device)
else:
    model = model.RNNModel(args.model, args.emsize, args.nout, args.nhid, args.nlayers, args.task,
                           args.dropout).to(device)

# for loss visualization
model_initial = copy.deepcopy(model)

print(f"{model.count_parameters()} parameters")

if args.model_path is not None:
    model = torch.load(args.model_path)

if args.task == "sequence_learning":
    criterion = nn.MSELoss()
if args.task == "sequence_classification":
    criterion = nn.BCELoss()


##############################  #################################################
# Training code
###############################################################################
def evaluate(dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_samples = 0
    total_correct = 0
    with torch.no_grad():
        for data, targets, masks in tqdm(dataloader, desc="Evaluating"):
            total_samples += targets.shape[0]
            if args.model == 'Transformer':
                output = model(data, masks)
                targets = targets
            else:
                output = model(data, masks)
            if args.task == "sequence_classification":
                total_correct += (((output >= 0.5) == targets.bool()).sum()).item()
            total_loss += criterion(output, targets).item()
    model.train()
    loss_per_sample = total_loss / len(dataloader) / args.batch_size
    total_accuracy = None
    if args.task == "sequence_learning":
        loss_per_sample = np.sqrt(loss_per_sample)
    elif args.task == "sequence_classification":
        total_accuracy = total_correct / total_samples
    return loss_per_sample, total_accuracy


def hessian_computation(model_final, criterion, loader, save_dir, split):
    hessian_comp = hessian(model_final,
                           criterion,
                           dataloader=loader,
                           cuda=False)

    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = np.mean(hessian_comp.trace())
    np.save(os.path.join(save_dir, f"hessian_{split}.npy"), np.array([top_eigenvalues[0], trace]))

    density_eigen, density_weight = hessian_comp.density()
    get_esd_plot(density_eigen, density_weight, save_dir, split)


def loss_landscape_viz(model_initial, model_final, criterion, loader_train, loader_test, save_dir):
    STEPS = 100
    MAX_DIST = 100
    CONTOUR_LEVELS = 50

    x, y, mask = iter(loader_train).__next__()
    metric = loss_landscapes.metrics.Loss(criterion, x, y, mask)

    x, y, mask = iter(loader_test).__next__()
    metric_test = loss_landscapes.metrics.Loss(criterion, x, y, mask)

    # interpolation
    loss_data = loss_landscapes.linear_interpolation(model_initial, model_final, metric, STEPS,
                                                     deepcopy_model=True)
    loss_data_test = loss_landscapes.linear_interpolation(model_initial, model_final, metric_test,
                                                          STEPS, deepcopy_model=True)
    plt.plot([1 / STEPS * i for i in range(STEPS)], loss_data, label="train")
    plt.plot([1 / STEPS * i for i in range(STEPS)], loss_data_test, label="test")
    plt.xlabel('Interpolation Coefficient')
    plt.title("Linear interpolation of loss")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_interpolation.pdf"))

    STEPS = 50

    def draw_contour_plots(metric_contour, split):
        # contour
        plt.figure()
        loss_data_fin = loss_landscapes.random_plane(model_final, metric_contour, MAX_DIST, STEPS,
                                                     normalization='layer', deepcopy_model=True)
        plt.contour(loss_data_fin, levels=CONTOUR_LEVELS)
        plt.title('Loss Contours around Trained Model')
        plt.savefig(os.path.join(save_dir, f"loss_contour_around_final_{split}.pdf"))

        # 3d landscape
        plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Surface Plot of Loss Landscape')
        plt.savefig(os.path.join(save_dir, f"loss_3d_around_final_{split}.pdf"))

    draw_contour_plots(metric, "train")
    draw_contour_plots(metric_test, "test")


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    best_val_loss = None

    train_iterator = trange(int(args.epochs), desc="Epoch")
    optimizer = AdamW(model.parameters(), lr=args.lr)
    for e, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(dataloader_train)
        for step, batch in enumerate(epoch_iterator):
            data, targets, masks = batch

            model.zero_grad()
            if args.model == 'Transformer':
                output = model(data, masks)
                targets = targets

            else:
                output = model(data, masks)
            loss = criterion(output, targets)
            loss.backward()
            clip_grad_value_(model.parameters(), args.clip)

            optimizer.step()

            total_loss += loss.item()

            if (step % args.log_interval == 0 and step > 0) or (
                    e == int(args.epochs) - 1 and step == len(epoch_iterator) - 1):
                val_loss, val_accuracy = evaluate(dataloader_val)
                if not best_val_loss or val_loss < best_val_loss:
                    with open(os.path.join(args.save, "model.pt"), 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                if args.task == "sequence_learning":
                    epoch_iterator.set_description(f"Average error: {val_loss}")
                elif args.task == "sequence_classification":
                    print(f"Average loss: {val_loss}")
                    epoch_iterator.set_description(f"Accuracy: {val_accuracy}")
            if args.dry_run:
                break
    return model


best_val_loss = None

final_model = train()
loss_landscape_viz(model_initial, final_model, criterion, dataloader_train, dataloader_test,
                   args.save)

hessian_computation(final_model, criterion, dataloader_test, args.save, "test")
hessian_computation(final_model, criterion, dataloader_train, args.save, "train")
