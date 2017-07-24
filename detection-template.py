#!/usr/bin/env python
# Basic template code to read the binetflow files of the Stratosphere project and try some classifiers
# First code done and shared by Jorge Pessoa dos Santos <J.G.PessoadosSantos@student.tudelft.nl>

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
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import argparse
import sys
from datetime import datetime

def read_dataset_folder(path):
    '''
    Gets a path folder, reads all the binetflow files on it and creates a panda dataset
    Just in case, the files name have to end with '*flow'
    '''
    datafiles = [filename for filename in listdir(path) if isfile(join(path, filename)) and not filename.startswith('.') and filename.endswith('flow')]
    datasets = []
    for filename in datafiles:
        print('Loading ' + filename + ' into the dataset.')
        dataset = pd.read_csv(join(path, filename), parse_dates=['StartTime'])
        # This is probably not needed since the StartTime is NOT unique
        dataset = dataset.set_index(dataset.StartTime)
        datasets.append(dataset)
    return pd.concat(datasets)

def make_categorical(dataset, cat):
    '''
    Convert one column to a categorical type
    '''
    # Converts the column to cotegorical
    dataset[cat] = pd.Categorical(dataset[cat])
    # Convert the categories to int
    dataset[cat] = dataset[cat].cat.codes
    return dataset

def process_features(dataset):
    ''' 
    Discards some features of the dataset and can create new.
    '''
    print('Processing features...')
    print('\tDiscarding column {}'.format('StartTime'))
    dataset = dataset.drop('StartTime', axis=1)
    #????
    dataset.reset_index()
    print('\tDiscarding column {}'.format('SrcAddr'))
    dataset = dataset.drop('SrcAddr', axis=1)
    print('\tDiscarding column {}'.format('DstAddr'))
    dataset = dataset.drop('DstAddr', axis=1)
    print('\tDiscarding column {}'.format('sTos'))
    dataset = dataset.drop('sTos', axis=1)
    print('\tDiscarding column {}'.format('dTos'))
    dataset = dataset.drop('dTos', axis=1)
    # Create categorical features
    dataset = make_categorical(dataset, 'Dir')
    dataset = make_categorical(dataset, 'Proto')
    # Convert the ports to categorical codes because some ports are not numbers. For exmaple, ICMP has ports with 0x03
    dataset = make_categorical(dataset, 'Sport')
    dataset = make_categorical(dataset, 'State')
    # Convert the ports to categorical codes because some ports are not numbers. For exmaple, ICMP has ports with 0x03
    dataset = make_categorical(dataset, 'Dport')
    print('Done')
    return dataset

def split_dataset(dataset):
    '''
    Separates the dataset in training data, training labels, testing data and testing labels. (Testing is cross-validation during training)
    '''
    y = dataset['Label']
    X = dataset.drop(['Label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_classifier(X, y):
    print('Training classifier')
    #clf = RandomForestClassifier(n_estimators = 50)
    clf = KNeighborsClassifier(3)
    #clf.fit_transform(X, y)
    clf.fit(X, y)
    print('Done')
    return clf

def test_classifier(clf, X, y):
    print('Testing classifier')
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    results = []
    for i in range(len(X)):
        pred = clf.predict(X[i:i+1])
        result = y[i:i+1][0]
        results.append(pred[0])
        if (pred == 'Normal' and result == 'Normal'):
            TN += 1
        if (pred == 'Normal' and result == 'Malware'):
            FN += 1
        if (pred == 'Malware' and result == 'Malware'):
            TP += 1
        if (pred == 'Malware' and result == 'Normal'):
            FP += 1
    # Compute and output the performance metrics. The last parameter is the name of the algor to store in the file
    compute_total_performance_metrics(TP, TN, FP, FN, str(type(clf)).split('\'')[1].split('.')[-1] )
    dataset = X
    dataset['Label'] = y
    dataset['Prediction'] = results
    return dataset

def compute_total_performance_metrics(TP, TN, FP, FN, algor_name):
    """ 
    Compute the performance metrics 
    """
    try:
        # or recall
        TPR = TP / float(TP + FN)
    except ZeroDivisionError:
        TPR = -1
    try:
        TNR = TN / float(TN + FP)
    except ZeroDivisionError:
        TNR = -1
    try:
        FPR = FP / float(TN + FP)
    except ZeroDivisionError:
        FPR = -1
    try:
        FNR = FN / float(TP + FN)
    except ZeroDivisionError:
        FNR = -1
    try:
        Precision = TP / float(TP + FP)
    except ZeroDivisionError:
        Precision = -1
    try:
        Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
    except ZeroDivisionError:
        Accuracy = -1
    try:
        ErrorRate = float(FN + FP) / float(TP + TN + FP + FN)
    except ZeroDivisionError:
        ErrorRate = -1
    # With beta=1 F-Measure is also Fscore and the balance is equal. beta=0.5 gives weights recall lower (FN less important)
    beta = 1.0
    try:
        FMeasure1 = ( ( (beta * beta) + 1 ) * Precision * TPR  ) / float( ( beta * beta * Precision ) + TPR)
    except ZeroDivisionError:
        FMeasure1 = -1
    # False Discovery Rate
    try:
        FDR = FP / float(TP + FP)
    except ZeroDivisionError:
        FDR = -1
    try:
        NPV = TN / float(TN + FN)
    except ZeroDivisionError:
        NPV = -1
    try:
        PLR = TPR / float(FPR)
    except ZeroDivisionError:
        PLR = -1
    try:
        NLR = FNR / float(TNR)
    except ZeroDivisionError:
        NLR = -1
    try:
        DOR = PLR / float(NLR)
    except ZeroDivisionError:
        DOR = -1

    f = open(outputfilename + '.metrics', 'w')
    print('Algorithm for training and testing: {}'.format(algor_name))
    f.write('Algorithm for training and testing: {}\n'.format(algor_name))
    print('Testing Dataset: {}'.format(args.data))
    f.write('Testing Dataset: {}\n'.format(args.data))
    print('TP = {}'.format(TP))
    f.write('TP = {}\n'.format(TP))
    print('FP = {}'.format(FP))
    f.write('FP = {}\n'.format(FP))
    print('TN = {}'.format(TN))
    f.write('TN = {}\n'.format(TN))
    print('FN = {}'.format(FN))
    f.write('FN = {}\n'.format(FN))
    print('TPR (Recall) = {}'.format(TPR))
    f.write('TPR (Recall) = {}\n'.format(TPR))
    print('TNR = {}'.format(TNR))
    f.write('TNR = {}\n'.format(TNR))
    print('FNR = {}'.format(FNR))
    f.write('FNR = {}\n'.format(FNR))
    print('FPR = {}'.format(FPR))
    f.write('FPR = {}\n'.format(FPR))
    print('Precision (Positive Prediction Value)= {}'.format(Precision))
    f.write('Precision (Positive Prediction Value)= {}\n'.format(Precision))
    print('Accuracy = {}'.format(Accuracy))
    f.write('Accuracy = {}\n'.format(Accuracy))
    print('ErrorRate = {}'.format(ErrorRate))
    f.write('ErrorRate = {}\n'.format(ErrorRate))
    print('False Discovery Rate = {}'.format(FDR))
    f.write('False Discovery Rate = {}\n'.format(FDR))
    print('Negative Predictive Value = {}'.format(NPV))
    f.write('Negative Predictive Value = {}\n'.format(NPV))
    print('Positive Likelihood Ratio (TPR/FPR)= {}'.format(PLR))
    f.write('Positive Likelihood Ratio (TPR/FPR)= {}\n'.format(PLR))
    print('Negative Likelihood Ratio (FNR/TNR)= {}'.format(NLR))
    f.write('Negative Likelihood Ratio (FNR/TNR)= {}\n'.format(NLR))
    print('Diagnostic odds Ratio (PLR/NLR)= {}'.format(NLR))
    f.write('Diagnostic odds Ratio (PLR/NLR)= {}\n'.format(NLR))
    print('FMeasure1 = {}'.format(FMeasure1))
    f.write('FMeasure1 = {}\n'.format(FMeasure1))
    f.close()

# Main
####################
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int)
    parser.add_argument('-e', '--debug', help='Amount of debugging. This shows inner information about the flows.', action='store', required=False, type=int)
    parser.add_argument('-t', '--training', help='Use the dataset stored in the folder Dataset for training, else use it for testing. You should first train something to store the model on disk.', action='store_true', required=False)
    parser.add_argument('-d', '--data', help='Use the dataset stored in this folder. The format is csv with the last column named \'Label\'.', action='store', required=True)
    parser.add_argument('-f', '--storeflows', help='Also create an output file with the original binetflows with a new column for the prediction.', action='store_true', required=False, default=False)
    parser.add_argument('-m', '--modelfilename', help='Name of the file where the model will be stored during training and read during testing.', action='store', required=False, type=str)

    args = parser.parse_args()

# Output file name
outputfilename = 'testing_results.' + datetime.now().strftime("%Y-%m-%d_%H:%M")

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
    if args.modelfilename:
        pickle.dump(clf, open( args.modelfilename, 'wb' ) )
    else:
        print 'No model file name provided, using the name of the classifier.'
        pickle.dump(clf, open( str(type(clf)).split('\'')[1].split('.')[-1]  + '-trained-model.pickle', 'wb' ) )
else:
    # Are we testing?
    print 'Testing the previous trained model on an testing dataset'
    print 'Loading model from disk...'
    try:
        #clf = pickle.load( open( './trained-model.pickle', 'rb' ))
        if args.modelfilename:
            clf = pickle.load( open( args.modelfilename, 'rb' ) )
        else:
            print 'No model file name provided. Please provide one.'
            sys.exit(-1)
    except IOError:
        print 'No model stored yet on disk. Please first train with a training dataset.'
        sys.exit(-1)

    # Separate the label from the dataset for the testing
    y = dataset['Label']
    X = dataset.drop(['Label'], axis=1)

    # Test
    dataset = test_classifier(clf, X, y)

    # Writes the classified dataset to disk
    if args.storeflows:
        dataset.to_csv(outputfilename + '.binetflow')
