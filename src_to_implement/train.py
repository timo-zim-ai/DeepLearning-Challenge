import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import csv
import os
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO

csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)


df = pd.read_csv(csv_path,sep=";")
df["labels_group"] = (df["crack"].astype(str)+"_"+df["inactive"].astype(str))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,stratify=df["labels_group"])

train_df = train_df.drop(columns="labels_group")
val_df = val_df.drop(columns="labels_group")

train_dataset = ChallengeDataset(train_df,"train")

val_dataset = ChallengeDataset(val_df,"val")


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

train_loader = t.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
val_loader = t.utils.data.DataLoader(dataset=val_dataset,batch_size=32,shuffle=True)


# create an instance of our ResNet model
# TODO
res = ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = nn.BCELoss()
optimizer = t.optim.Adam(res.parameters(),lr = 1e-3, weight_decay=1e-4)

trainer = Trainer(model=res,crit=criterion,optim=optimizer,train_dl=train_loader,val_test_dl=val_loader,early_stopping_patience=8)
# go, go, go... call fit on trainer
res = trainer.fit(50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')