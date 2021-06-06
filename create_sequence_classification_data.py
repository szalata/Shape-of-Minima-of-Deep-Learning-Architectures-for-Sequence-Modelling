import argparse
import os

import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    Predict whether the sum of the sequence exceeds the threshold
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=5, help='length of the sequence/sample')
    parser.add_argument('--samples', type=int, default=5000, help='the number of samples to return')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--max_number', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_fraction', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, help='name of the file to write',
                        default="data/sequence_classification")
    args = parser.parse_args()
    np.random.seed(args.seed)

    out_array = np.random.uniform(
        -args.max_number, high=args.max_number, size=(args.samples, args.seq_len))
    targets_array = out_array.sum(axis=1) >= args.threshold

    # create data splits
    X_train, X_test, y_train, y_test = train_test_split(out_array, targets_array,
                                                        test_size=args.test_fraction,
                                                        random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=args.test_fraction,
                                                      random_state=args.seed)
    # create directory if doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    split_data_mapping = {"train": (X_train, y_train),
                          "val": (X_val, y_val),
                          "test": (X_test, y_test)}
    for split in ["train", "val", "test"]:
        X, y = split_data_mapping[split]
        np.save(os.path.join(args.output_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(args.output_dir, f"y_{split}.npy"), y)
