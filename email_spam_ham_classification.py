# I accidently uploaded this file twice and I don't know how to remove it, so it will stay here...

from bayes_classifier_v2 import *

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Any, Set, Dict
from collections import Counter

# Data loading and processing stuff (emails -> vectors, idk what's going on here)

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



if __name__ == '__main__':
    data_root_path = './data/'
    dataset_name = 'enron_spam_data.csv'
    (x_train, y_train), (x_test, y_test) = load_and_process_enron_data(data_root_path, dataset_name, num_features=200)

    print(sum(y_train))

    y_train = np.array([[1,0] if y_train[k]==0 else [0,1] for k in range(len(y_train))])

    l, p = get_model(x_train, y_train,[0,1])



    for k in range(10):
        result = map_classify(x_test[k], [0, 1], l, p)
        print(result[0], y_test[k])
        print(result[1])


    # confusion matrix for testing
    conf = np.zeros((2, 2))

    for k in range(len(x_test)):
        y_pred = map_classify(x_test[k], [0,1], l, p)[0]
        y = y_test[k]

        conf[y][y_pred] += 1

    print("Confusion Matrix on Test Data:")
    print(conf)
