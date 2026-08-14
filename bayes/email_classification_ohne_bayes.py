import torch

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Any, Set, Dict
from collections import Counter
from math import ceil, floor

from pip._internal import req

# mal wieder geklaut

SPAM_KEY = 'spam'
NO_SPAM_KEY = 'ham'
LABELS = [SPAM_KEY, NO_SPAM_KEY]

def extract_feature(email, tokens) -> np.ndarray:
    return np.array([int(token in email) for token in tokens])

def binarise_labels(labels: np.ndarray) -> np.ndarray:
    """
        1 = spam
        0 = ham
    """
    return np.array([int(item == LABELS[0]) for item in labels])

def tokenise(
        message: str,
        min_length: int=4
) -> Set :
    words = re.findall(r"[a-z0-9]+", message.lower())
    return set([item for item in words if len(item) >= min_length])

def load_and_process_enron_data(
        root_path: str,
        file_name: str,
        test_split: float=0.2,
        num_features: int=200
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(root_path, file_name))
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['Spam/Ham'] = df['Spam/Ham'].fillna('')

    subjects = df['Subject']
    messages = df['Message']
    labels = df['Spam/Ham']
    emails = subjects + ' ' + messages      # merge subject and message to email

    email_train, email_test, labels_train, labels_test = train_test_split(
        emails, labels, test_size=test_split, random_state=123, stratify=labels
    )

    # --- extract features
    counter = Counter()
    for item in email_train:
        counter.update(tokenise(item))
    top_tokens = [w for w, _ in counter.most_common(num_features)]
    features_train = np.array([extract_feature(email, top_tokens) for email in email_train])
    features_test = np.array([extract_feature(email, top_tokens) for email in email_test])

    # --- generate features
    bin_labels_train = binarise_labels(labels_train)
    bin_labels_test = binarise_labels(labels_test)

    return (features_train, bin_labels_train), (features_test, bin_labels_test)






def round_threshold(x, threshold):
    f = floor(x)
    c = ceil(x)
    d = x - c + 1
    if d < threshold:
        return f
    else:
        return c

A = torch.randn((10,200), requires_grad=True)
b = torch.randn((10,1), requires_grad=True)
c = torch.randn((1,10), requires_grad=True)

opt = torch.optim.SGD([A, b, c], lr=0.001)


def forward(x):
    return 1 / (1 + torch.exp(-torch.matmul(c/100, torch.relu(torch.matmul(A/100, x.transpose(1,0)) + b/100))))

def log_likelihood(x,y):
    return - torch.sum(y_train*torch.log(forward(x_train)) + (1-y_train)*torch.log(1-forward(x_train)))

def train(x,y, epochs):
    for epoch in range(epochs):
        opt.zero_grad()
        loss = log_likelihood(x,y)
        loss.backward()
        opt.step()
        print(epoch, loss.item())

def save_model(n):
    data = {"A":A, "b":b, "c":c}
    torch.save(data, "saved_models/model{}.pt".format(n))

def load_model(n):
    data = torch.load("saved_models/model{}.pt".format(n))
    global A, b, c
    A = data["A"]
    b = data["b"]
    c = data["c"]


if __name__ == "__main__":
    data_root_path = './data/'
    dataset_name = 'enron_spam_data.csv'
    (x_train, y_train), (x_test, y_test) = load_and_process_enron_data(data_root_path, dataset_name, num_features=200)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    train(x_train, y_train,25000)

    save_model(2)



    for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        # confusion matrix for testing
        conf = np.zeros((2, 2))

        for k in range(len(x_test)-1):
            y_pred = round_threshold(forward(x_test[k:k+1]).detach().item(), a)
            y = int(y_test[k].item())

            conf[y][y_pred] += 1

        print(a)
        print("Confusion Matrix on Test Data:")
        print(conf)
        print()






