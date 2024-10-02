#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-10-02
Description: Helper methods to interact with the dataset quickly.
"""

# Import statements
import os
import re
import pandas as pd
from random import randint
import pickle
from unidecode import unidecode
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

SEPARATOR = 100 * '-'


def save_to_pickle(filename: str, data: pd.Series) -> None:
    """Pickle a dataset to disk for future use

    Args:
        filename (str): file path to save the data
        data (pd.Series): dataset to save
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(filename: str) -> pd.Series:
    """Loads a pickled dataset from disk to avoid re-importing it

    Args:
        filename (str): file path to load the data from

    Returns:
        pd.Series: dataset loaded from the pickle file
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pickled_data_exists(filenames: list) -> bool:
    """Checks if all the files in the list exist

    Args:
        filenames (list): list of file paths to check

    Returns:
        bool: True if all files exist, False otherwise
    """
    return all(os.path.exists(filename) for filename in filenames)


def print_text(X, y, idx=None):
    try:
        print(SEPARATOR)
        if idx is None:
            idx = randint(0, X.shape[0])
        print(f"[{idx}]", X[idx], "-->", y[idx])
        print(SEPARATOR)

    except Exception as e:
        print(f"Can't print email contents: {e}")


def clean_text(text):

    # Remove special characters
    text = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñ\s]', '', text)

    # Normalize accents
    text = unidecode(text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower().strip()

    return text


def remove_stopwords(text):

    # get spanish stopwords
    stopwords_sp = stopwords.words("spanish")

    # tokenize the text by splitting on spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_sp]

    return ' '.join(tokens)


def clean_dataset(X):

    # apply cleaning
    X_clean = X.progress_apply(clean_text)

    # apply removing stopwords
    X_clean = X_clean.progress_apply(remove_stopwords)

    return X_clean


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

    # reset indexes
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def load_dataset(file_path, force_reload=False):

    # check if pickle files exist
    x_train_pickle = os.path.join('pickle/x_train.pkl')
    y_train_pickle = os.path.join('pickle/y_train.pkl')
    x_test_pickle = os.path.join('pickle/x_test.pkl')
    y_test_pickle = os.path.join('pickle/y_test.pkl')

    if pickled_data_exists([x_train_pickle, y_train_pickle, x_test_pickle, y_test_pickle]) and not force_reload:

        # Load pickled data
        print("Loading split dataset from pickle files")
        X_train = load_from_pickle(x_train_pickle)
        X_test = load_from_pickle(x_test_pickle)
        y_train = load_from_pickle(y_train_pickle)
        y_test = load_from_pickle(y_test_pickle)

    else:
        print("No pickle file found. Loading and cleaning dataset.")

        # load dataset
        data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
        X, y = data["text"], data["label"]

        # clean dataset
        X_clean = clean_dataset(X)

        # split dataset
        X_train, X_test, y_train, y_test = split_dataset(X_clean, y)

        # save pickle files
        os.makedirs("pickle", exist_ok=True)
        save_to_pickle(x_train_pickle, X_train)
        save_to_pickle(x_test_pickle, X_test)
        save_to_pickle(y_train_pickle, y_train)
        save_to_pickle(y_test_pickle, y_test)

    return X_train, X_test, y_train, y_test
