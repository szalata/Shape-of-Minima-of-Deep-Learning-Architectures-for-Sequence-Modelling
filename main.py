# coding: utf-8
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import model
from SeqDataset import SeqDataset

parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=1,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=256,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=5,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence length')
parser.add_argument('--nout', type=int, default=1,
                    help='number of outputs')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--data_dir', type=str, default='data/sequence_learning',
                    help='directory with the dataset')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()

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
dataloader_train = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)
dataloader_val = DataLoader(dataset_val,
                            shuffle=False,
                            batch_size=args.batch_size)

if args.model == 'Transformer':
    model = model.TransformerModel(args.emsize, args.nout, args.nhead, args.nhid, args.nlayers,
                                   args.dropout).to(device)
else:
    model = model.RNNModel(args.model, args.emsize, args.nout, args.nhid, args.nlayers,
                           args.dropout).to(device)

criterion = nn.MSELoss()


##############################  #################################################
# Training code
###############################################################################
def evaluate(dataloader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            if args.model == 'Transformer':
                output = model(data)
            else:
                output = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(dataloader) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    train_iterator = trange(int(args.epochs), desc="Epoch")
    optimizer = AdamW(model.parameters())#, lr=args.learning_rate, eps=args.adam_epsilon)
    for _ in train_iterator:
        epoch_iterator = tqdm(dataloader_train)
        for step, batch in enumerate(epoch_iterator):
            data, targets = batch
            model.zero_grad()
            if args.model == 'Transformer':
                output = model(data).squeeze()
            else:
                output = model(data, hidden)[:, -1].squeeze(dim=1)
            loss = criterion(output, targets)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if step % args.log_interval == 0 and step > 0:
                cur_loss = total_loss / args.log_interval
                epoch_iterator.set_description(f"Loss: {cur_loss}")
                total_loss = 0
            if args.dry_run:
                break


# Loop over epochs.
lr = args.lr
best_val_loss = None

train()
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (
                time.time() - epoch_start_time),
                                                                                     val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 1.
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | '.format(
    test_loss))
print('=' * 89)
