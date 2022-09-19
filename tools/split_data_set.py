import pandas as pd
import numpy as np

df = pd.read_csv('../data/original.csv')

np.random.seed(42)

msk = np.random.rand(len(df)) < 0.8

train = df[msk]

test = df[~msk]

train.to_csv('../data/train.csv', index=False)
test.to_csv('../data/test.csv', index=False)
