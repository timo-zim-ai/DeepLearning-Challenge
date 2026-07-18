
import pandas as pd
from data import ChallengeDataset
import torch as t
import numpy as np
df = pd.read_csv("src_to_implement\data.csv",sep=";")
dataset = ChallengeDataset(data=df,mode='string')


df = pd.read_csv("src_to_implement\data.csv",sep=";")

X = df["filename"].tolist()

y1 = df["crack"].tolist()
y2 = df["inactive"].tolist() 
print(np.array(X))
print(np.array(y1))
print(np.array(y2))