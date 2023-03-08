import torch

# This script was written by Leo Konle

system = "wcph113"
if system == "wcph113":
    import sys

    sys.path.append('/mnt/data/users/schroeter/PyNovellaHistory')

# Legt fest welche und wie viele GPU verwendet werden soll
# Für alle GPUs: os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# Prüfen welche GPUs frei sind: nvidia-smi


from preprocessing.presetting import local_temp_directory

from transformers import BertTokenizerFast, BertModel, BertConfig, BertForTokenClassification, TextDataset
from transformers import AdamW, PreTrainedTokenizer, PreTrainedTokenizerFast, BertForSequenceClassification, \
    get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup, Trainer, DataCollatorForTokenClassification, AdamWeightDecay
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
import torch
from tqdm.notebook import tqdm
import os
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
import json
from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random
import re
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pickle as pkl

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(torch.cuda.is_available())

# In[ ]:


# Parameter
modelname = "deepset/gbert-large"  # huggingface modelname
bsize = 2  # integer
epochs = 10  # integer (Dauer des Trainings)
learningrate = 5e-4  # float (gebräuchlich sind Werte zwischen 1e-4 und 1e-5)

# In[ ]:


model = BertForSequenceClassification.from_pretrained(modelname, torchscript=True, num_labels=2)


# num_labels: Anzahl der Klassen


# In[ ]:


def inner_shuffle_train(X_train, y_train, bsize, state):
    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1, random_state=state)
    class_size = min(Counter(y_train).values())

    sel = pd.concat([frame[frame["Y"] == 0].iloc[:class_size, :],
                     frame[frame["Y"] == 1].iloc[:class_size, :]],
                    axis=0).sample(frac=1, random_state=state)

    X = np.array(sel.iloc[:, :512])
    y = list(sel["Y"])

    num_examples = len(X)
    i = 0
    inputs = []
    labels = []
    batches = []

    while i != num_examples:

        inputs.append(torch.tensor(X[i]))
        labels.append(torch.tensor(y[i]))

        if len(inputs) == bsize:
            batches.append([inputs, labels])
            inputs = []
            labels = []
        i += 1

    return batches

def arrange_val_batches(X_train, y_train, bsize):
    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1)
    class_size = min(Counter(y_train).values())

    sel = pd.concat([frame[frame["Y"] == 0].iloc[:class_size, :],
                     frame[frame["Y"] == 1].iloc[:class_size, :]],
                    axis=0).sample(frac=1)

    X = np.array(sel.iloc[:, :512])
    y = list(sel["Y"])
    num_examples = len(X)

    i = 0
    inputs = []
    labels = []
    batches = []

    while i != num_examples:

        inputs.append(torch.tensor(X[i]))
        labels.append(torch.tensor(y[i]))

        if len(inputs) == bsize:
            batches.append([inputs, labels])
            inputs = []
            labels = []
        i += 1

    return batches

# Trainings- und Testdaten laden
X = pkl.load(open(os.path.join(local_temp_directory(system), "X.pkl"), "rb"))  # Token
Y = pkl.load(open(os.path.join(local_temp_directory(system), "Y.pkl"), "rb"))  # Label
M = pkl.load(open(os.path.join(local_temp_directory(system), "M.pkl"), "rb"))  # Maskierung

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
val_batches = arrange_val_batches(X_test, y_test, bsize)

PATH = "/mnt/data/users/schroeter/PyNovellaHistory/scripts_novellas/deeplearning/mymodel_0.94.pt"

# Model class must be defined somewhere

modelname = "deepset/gbert-large" # huggingface modelname
model = BertForSequenceClassification.from_pretrained(modelname, torchscript=True, num_labels=2)
device = "cuda"
model.to(device)
#Zeile ausführen für multi-gpu support
model = nn.DataParallel(model)


model.state_dict(torch.load(PATH))
print(model)

t = []
p = []
with torch.no_grad():
    # Validation Loop
    for batch in val_batches:
        b_input_ids = torch.stack(batch[0]).to(device)
        b_labels = torch.stack(batch[1]).to(device)

        outputs = model(b_input_ids, token_type_ids=None)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        tlabel = np.argmax(logits, axis=1)
        label_ids = b_labels.to('cpu').numpy()
        trues = label_ids.flatten()
        pred = tlabel.flatten()

        t.append(trues)
        p.append(pred)

report = classification_report(np.stack(t).flatten(), np.stack(p).flatten())
print(report)
