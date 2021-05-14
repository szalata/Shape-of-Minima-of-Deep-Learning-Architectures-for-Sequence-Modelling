import torch 
import torch.nn as nn
from TransformerModel import TransformerModel
from FibonacciDataset import FibonacciDataset
from torch.utils.data import DataLoader
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, default=32)
    
    parser.add_argument('nb_epochs', type=int, default=100)
    
    args = parser.parse_args()
        

    # get datasets and dataloaders
    train_dataset = FibonacciDataset("FibData_train.csv")
    test_dataset = FibonacciDataset("FibData_test.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    
    model = torch.nn.Transformer()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters())

    for e in range(args.nb_epochs):
        
        for feature, label in train_dataloader:
            
            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, label)
            loss.backward()

        with torch.no_grad():
            test_loss = 0
            for feature, label in test_dataloader:
                test_pred = model(feature)
                test_loss += criterion(pred, label)
            print(test_loss)
