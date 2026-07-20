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

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score, recall_score
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

trainer = Trainer(cuda=False,model=res,crit=criterion,optim=optimizer,train_dl=train_loader,val_test_dl=val_loader,early_stopping_patience=30)

ckp = t.load("src_to_implement\checkpoints\checkpoint_047.ckp", map_location="cpu")
print(ckp.keys())

trainer._model.load_state_dict(ckp["state_dict"])

trainer.restore_checkpoint(47)

trainer._model.eval()

all_outputs = []
all_labels = []

thresholds=(0.5,0.28)

with t.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(trainer.device)
        labels = labels.to(trainer.device)

        outputs = trainer._model(inputs)

        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

all_outputs = t.cat(all_outputs, dim=0)
all_labels = t.cat(all_labels, dim=0).int()

thresholds = t.tensor(thresholds)
all_predictions = (all_outputs >= thresholds).int()

class_f1 = f1_score(
    all_labels.numpy(),
    all_predictions.numpy(),
    average=None,
    zero_division=0
)

macro_f1 = class_f1.mean()

print(f"Crack F1:    {class_f1[0]:.4f}")
print(f"Inactive F1: {class_f1[1]:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")

precision = precision_score(
    all_labels.numpy(),
    all_predictions.numpy(),
    average=None,
    zero_division=0
)

recall = recall_score(
    all_labels.numpy(),
    all_predictions.numpy(),
    average=None,
    zero_division=0
)

print("Crack Precision:", precision[0])
print("Crack Recall:", recall[0])

print("Inactive Precision:", precision[1])
print("Inactive Recall:", recall[1])

y_true = all_labels.numpy()
y_prob = all_outputs.numpy()

best_threshold = None
best_f1 = -1

for threshold in np.arange(0.05, 0.96, 0.01):
    pred_inactive = (y_prob[:, 0] >= threshold).astype(int)

    f1 = f1_score(
        y_true[:, 1],
        pred_inactive,
        zero_division=0
    )

    print(f"{threshold:.2f} -> {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("Best inactive threshold:", best_threshold)
print("Best inactive F1:", best_f1)

best_macro = -1
best_pair = None

for crack_threshold in np.arange(0.1, 0.91, 0.01):
    for inactive_threshold in np.arange(0.1, 0.91, 0.01):

        thresholds = np.array([
            crack_threshold,
            inactive_threshold
        ])

        predictions = (y_prob >= thresholds).astype(int)

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_pair = (
                crack_threshold,
                inactive_threshold
            )

print("Best thresholds:", best_pair)
print("Best macro F1:", best_macro)