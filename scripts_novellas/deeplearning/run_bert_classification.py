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
from transformers import AdamW, PreTrainedTokenizer, PreTrainedTokenizerFast, BertForSequenceClassification,get_cosine_schedule_with_warmup
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
from sklearn.model_selection import  train_test_split
import random
import re
import argparse
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import pickle as pkl


os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(torch.cuda.is_available())

# In[ ]:


# Parameter
modelname = "deepset/gbert-large" # huggingface modelname
bsize = 2 # integer
epochs = 10 # integer (Dauer des Trainings)
learningrate = 5e-4 # float (gebräuchlich sind Werte zwischen 1e-4 und 1e-5)


# In[ ]:


model = BertForSequenceClassification.from_pretrained(modelname, torchscript=True, num_labels=2)
# num_labels: Anzahl der Klassen


# In[ ]:


def inner_shuffle_train(X_train, y_train, bsize, state):

    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1,random_state=state)
    class_size = min(Counter(y_train).values())
    
    sel = pd.concat([frame[frame["Y"] == 0].iloc[:class_size,:], 
                     frame[frame["Y"] == 1].iloc[:class_size,:]], 
                    axis=0).sample(frac=1,random_state=state)
    
    X = np.array(sel.iloc[:,:512])
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
            batches.append([inputs,labels])
            inputs = []
            labels = []
        i+=1

    return batches


# In[ ]:


def arrange_val_batches(X_train, y_train, bsize):


    frame = pd.DataFrame(X_train)
    frame["Y"] = y_train
    frame = frame.sample(frac=1)
    class_size = min(Counter(y_train).values())
    
    sel = pd.concat([frame[frame["Y"] == 0].iloc[:class_size,:], 
                     frame[frame["Y"] == 1].iloc[:class_size,:]], 
                    axis=0).sample(frac=1)
    
    X = np.array(sel.iloc[:,:512])
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
            batches.append([inputs,labels])
            inputs = []
            labels = []
        i+=1

    return batches


# In[ ]:


X = pkl.load(open(os.path.join(local_temp_directory(system),"X.pkl"),"rb")) # Token
Y = pkl.load(open(os.path.join(local_temp_directory(system),"Y.pkl"),"rb")) # Label
M = pkl.load(open(os.path.join(local_temp_directory(system),"M.pkl"),"rb")) # Maskierung
# Trainingsdaten laden


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# train test split
batches = inner_shuffle_train(X_train, y_train, bsize, 43)
val_batches = arrange_val_batches(X_test, y_test, bsize)
# batch creation


# In[ ]:


optimizer = AdamW(model.parameters(),lr = learningrate)
total_steps = len(batches) * epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


# In[ ]:


device = "cuda"
model.to(device)

#Zeile ausführen für multi-gpu support
model = nn.DataParallel(model)

print("Train...")
report = []
best_score = 0.5


# In[ ]:


# Training Loop
for epoch_i in range(0, epochs):

    batches = inner_shuffle_train(X_train,y_train, bsize, epoch_i*10)
    print("Epoch: "+str(epoch_i))
    print(len(batches))
    total_loss = 0
    model.train()
    i = 0
    f1 = 0
    rep_t = []
    rep_p = []
    for step, batch in enumerate(batches):
        i+=1

        b_input_ids = torch.stack(batch[0]).to(device)
        b_labels = torch.stack(batch[1]).to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None,labels=b_labels)

        loss = outputs[0].sum()
        total_loss += loss.item()


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.9)
        
        optimizer.step()
        scheduler.step()
        avg_train_loss = total_loss / i

        if i % 5 == 0:
            # Loss report
            print("Batch Loss: "+str(i)+" "+str(avg_train_loss))

    t = []
    p = []
    
    model.eval()
    
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
    f1_epoch = report.split("\n")[-3]
    f1_epoch = float(re.sub("\s+"," ",f1_epoch).split(" ")[-2])
    
    if f1_epoch >= best_score:
        torch.save(model.state_dict(), "mymodel_"+str(f1_epoch)+".pt")

    model.train()

