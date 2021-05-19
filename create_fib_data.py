import pandas as pd
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_len', type=int, default=100, help='length of the fibonacci sequence')
    parser.add_argument('type', type=str, default='train', help='train/test')
    args = parser.parse_args()

    prev1 = 0
    prev2 = 1
    curr = 1

    data = {prev1}

    l = [prev1, prev2, curr]
    for i in range(args.seq_len - 1):
        tmp = curr
        prev1 = prev2
        prev2 = tmp
        curr = prev1 + prev2
        l.append(curr)

    df = pd.DataFrame(l)
    np.savetxt(f'FibData/{args.type}.txt', l, fmt='%d', newline=' ')
