import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Any, Set, Dict
from collections import Counter

# ===================================================================
# DATA HANDLING
# ===================================================================

# --- Enron dataset -------------------------------------------------

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

# --- Your custom dataset -------------------------------------------

def load_and_process_custom_data(
        root_path: str,
        file_name: str,
        test_split: float = 0.2,
        **kwargs: Dict[str, Any]
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:

    # TODO: implement me ...

    features_train = ...
    labels_train = ...
    features_test = ...
    bin_labels_test = ...

    return (features_train, labels_train), (features_test, bin_labels_test)

# ===================================================================
# BAYES CLASSIFICATION
# ===================================================================

def estimate_prior(labels: np.ndarray) -> np.ndarray:
    prior = np.zeros(2)

    # TODO: implement me ...

    return prior

def estimate_likelihood(
        feature_arr: np.ndarray,
        labels: np.ndarray,
        laplace_smoothing: bool=False
) -> np.ndarray:
    """
        Function which approximates the likelihood function.

        NOTE
        ----
            > Try to renounce on nested for loops - makes the implementation very slow. Use
                broadcasting instead

    """
    num_features = feature_arr.shape[1]
    likelihood = np.zeros((2, num_features, 2))

    # TODO: implement me ...

    return likelihood

def predict(
        x: np.ndarray,
        prior: np.ndarray,
        likelihood: np.ndarray
) -> np.ndarray:
    log_prior = np.log(prior)
    log_likelihood = np.log(likelihood)
    scores = np.zeros((x.shape[0], 2))

    # TODO: implement me ...

    return ...

# ===================================================================
# EVALUATION
# ===================================================================

def determine_confusion_matrix(
        y_test: np.ndarray,
        y_pred: np.ndarray
) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_test, y_pred, normalize='true')

def main():
    data_root_path = './data/'
    dataset_name = 'enron_spam_data.csv'
    (x_train, y_train), (x_test, y_test) = load_and_process_enron_data(data_root_path, dataset_name, num_features=200)

    prior = estimate_prior(y_train)
    likelihood = estimate_likelihood(x_train, y_train, laplace_smoothing=True)

    y_pred = predict(x_test, prior, likelihood)
    conf = determine_confusion_matrix(y_test, y_pred)

    print(conf)

if __name__ == '__main__':
    main()