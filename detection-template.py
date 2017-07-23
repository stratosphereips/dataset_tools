#!/usr/bin/env python
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
try:
    # For linux use model_selection
    from sklearn.model_selection import train_test_split
except ImportError:
    # For MACOS use cross_validation, since model_selection is not in the repos yet
    from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse
import sys

def read_dataset_folder(path):
    """
    Gets a path folder, reads all the binetflow files on it and creates a panda dataset
    Just in case, the files name have to end with '*flow'
    """
    datafiles = [filename for filename in listdir(path) if isfile(join(path, filename)) and not filename.startswith(".") and filename.endswith('flow')]
    datasets = []
    for filename in datafiles:
        print("Loading " + filename + "...")
        dataset = pd.read_csv(join(path, filename), parse_dates=["StartTime"])
        # This is controversial
        dataset = dataset.set_index(dataset.StartTime)
        datasets.append(dataset)
        print("Done")
    return pd.concat(datasets)

def make_categorical(dataset, cat):
    """
    ???
    """
    dataset[cat] = pd.Categorical(dataset[cat])
    dataset[cat] = dataset[cat].cat.codes
    return dataset

def process_features(dataset):
    """ 
    Discards some features of the dataset and can create new.
    """
    print("Processing features...")
    print('\tDiscarding column {}'.format('StartTime'))
    dataset = dataset.drop(["StartTime"], axis=1)
    #????
    dataset.reset_index()
    print('\tDiscarding column {}'.format('SrcAddr'))
    dataset = dataset.drop(["SrcAddr"], axis=1)
    print('\tDiscarding column {}'.format('DstAddr'))
    dataset = dataset.drop(["DstAddr"], axis=1)
    print('\tDiscarding column {}'.format('sTos'))
    dataset = dataset.drop(["sTos"], axis=1)
    print('\tDiscarding column {}'.format('dTos'))
    dataset = dataset.drop(["dTos"], axis=1)
    # Create categorical features
    dataset = make_categorical(dataset, "Dir")
    dataset = make_categorical(dataset, "Proto")
    # ???
    dataset = make_categorical(dataset, "Sport")
    dataset = make_categorical(dataset, "State")
    # ???
    dataset = make_categorical(dataset, "Dport")
    print("Done")
    return dataset

def split_dataset(dataset):
    """
    Separates the dataset in training data, training labels, testing data and testing labels. (Testing is cross-validation during training)
    """
    y = dataset["Label"]
    X = dataset.drop(["Label"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_classifier(X, y):
    print("Training classifier")
    clf = RandomForestClassifier(n_estimators = 50)
    #clf.fit_transform(X, y)
    clf.fit_transform(X, y)
    print("Done")
    return clf

def test_classifier(clf, X, y):
    print("Testing classifier")
    tp, fp, tn, fn = 0, 0, 0, 0

    results = []
    for i in range(len(X)):
        pred = clf.predict(X[i:i+1])
        result = y[i:i+1][0]
        results.append(pred[0])

        if(pred == "Normal" and result == "Normal"):
            tn += 1
        if(pred == "Normal" and result == "Malware"):
            fn += 1
        if(pred == "Malware" and result == "Malware"):
            tp += 1
        if(pred == "Malware" and result == "Normal"):
            fp += 1

    print("TP = " + str(tp) + " | FP = " + str(fp))
    print("TN = " + str(tn) + " | FN = " + str(fn))
    if((tp+fp != 0) and (tp+fn != 0)):
        print("Precision = " + str(tp/(tp+fp)))
        print("Recall = " + str(tp/(tp+fn)))
    print("Done")
    
    dataset = X
    dataset["Label"] = y
    dataset["Prediction"] = results
    return dataset


# Main
####################
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int)
    parser.add_argument('-e', '--debug', help='Amount of debugging. This shows inner information about the flows.', action='store', required=False, type=int)
    parser.add_argument('-t', '--training', help='Use the dataset stored in the folder Dataset for training, else use it for testing. You should first train something to store the model on disk.', action='store_true', required=False)
    parser.add_argument('-d', '--data', help='Use the dataset stored in this folder. The format is csv with the last column named \'Label\'.', action='store', required=True)
    args = parser.parse_args()

# Loads binetflow files from the folder as training dataset
dataset = read_dataset_folder(args.data)

# Process features
dataset = process_features(dataset)

# Are we training?
if args.training:
    # Split in training/CV
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    # Train a classifier
    clf = train_classifier(X_train, y_train)

    # Store the model on disk
    print 'Storing the model on disk'
    pickle.dump(clf, open( "./trained-model.pickle", "wb" ) )
else:
    # Are we testing?
    print 'Testing the previous trained model on an testing dataset'
    print 'Loading model from disk...'
    try:
        clf = pickle.load( open( "./trained-model.pickle", "rb" ))
    except IOError:
        print 'No model stored yet on disk. Please first train with a training dataset.'
        sys.exit(-1)

    # Get the label column
    y = dataset["Label"]
    X = dataset.drop(["Label"], axis=1)

    # Test
    dataset = test_classifier(clf, X, y)

    # Writes the classified dataset to disk
    dataset.to_csv("results.labeled..binetflow")
