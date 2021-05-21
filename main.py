# coding: utf-8
import argparse
import time
import torch
import torch.nn as nn

from SeqDataset import SeqDataset
from torch.utils.data import DataLoader
import model

parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--data', type=str, default='./FibData/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=10,
                    help='size of word embeddings (sequence length in our case?)')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=10,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=4,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model2.pt',
                    help='path to save the final model')
parser.add_argument('--nhead', type=int, default=10,
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
train_data = SeqDataset("data", "train")
test_data = SeqDataset("data", "test")
val_data = SeqDataset("data", "val")

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
###############################################################################
# Build the model
###############################################################################


if args.model == 'Transformer':
    model = model.TransformerModel(args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)


criterion = nn.MSELoss() #nn.L1Loss()

lr = 0.2# learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
###############################################################################
# Training code
###############################################################################

def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
   
  
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            
            if args.model == 'Transformer':
                output = model(X.float()).squeeze(dim=1)
                #output = output.view(-1, ntokens)
            
            total_loss += len(X) * criterion(output, y).item()
    return total_loss / (len(test_loader) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
   
   
    for batch, (X, y) in enumerate(train_loader):

        optimizer.zero_grad()
        
        output = model(X.float()).squeeze(dim=1)

        loss = criterion(output, y.float())
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, batch, len(train_data),scheduler.get_last_lr()[0],
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                           val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
    
        scheduler.step()

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
test_loss = evaluate()
print('=' * 89)
print('| End of training | test loss {:5.2f} | '.format(
    test_loss))
print('=' * 89)
