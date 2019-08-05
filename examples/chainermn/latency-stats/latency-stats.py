#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd


def try_read(path, rank):
    filename = os.path.join(path, "latency.{}.log".format(rank))
    print('Reading', filename, '...', end='')
    df = pd.read_csv(filename, header=[0]) #names=[str(rank)])
    print('found.')
    return df

def main():
    '''Computes the elapsed time between two events (in milliseconds with
    a resolution of around 0.5 microseconds).

    https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="directory to read", default='result')
    parser.add_argument('--out', help="filename to save", default='result.csv')
    # TODO(kuenishi): Add argument to ignore several ranks where files may lack
    args = parser.parse_args()

    rank = 0
    dfs = []
    while True:
        try:
            df = try_read(args.path, rank)
            dfs.append(df)
            rank += 1
        except FileNotFoundError:
            print('not found.')
            break
    df = pd.concat(dfs, axis=1)

    ext = os.path.splitext(args.out)[1]

    if ext == '.pq':
        df.to_parquet(args.out)
        print("Saved to", args.out)
    elif ext == '.csv':
        df.to_csv(args.out, index=False)
        print("Saved to", args.out)
    else:
        print(df)

    sys.exit(0)

if __name__ == '__main__':
    main()
