import pandas as pd
import argparse
import numpy as np





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int, default=100, help='length of the sequence')
    parser.add_argument('length', type=int, default=5, help='length of the sequence')
    parser.add_argument('type', type=str, default='train', help='train/test')
    args = parser.parse_args()


    l = []
    for i in range(args.size-1):
        curr = [j+i for j in range(args.length)]
        if i%3==0:
            curr[0] = curr[-1]
        elif i%3==1:
            curr[0]= curr[-1] = curr[-2]
        else:
            curr[0]= curr[-1] = curr[-2] = curr[-3]
        l.append(curr)

       
    df = pd.DataFrame(l)
    np.savetxt(f'MemData/{args.type}.txt', l, fmt='%d', newline='\n')
    