import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
import random
'''
FUNCTIONS TO SPLIT DF INTO TRAIN AND TEST SETS
'''
#selects specific year ranges for test and train data
def select_years(train, test, train_years, test_years):
    drops = []
    #goes through dataset and drops rows that aren't in train year range
    for i, row in train.iterrows():
      year = train.at[i,'GSS YEAR FOR THIS RESPONDENT                       ']
      if int(year) not in train_years:
        drops.append(i)
    train = train.drop(drops)
    drops = []
    #goes through dataset and drops rows that aren't in test year range
    for i, row in test.iterrows():
      year = test.at[i,'GSS YEAR FOR THIS RESPONDENT                       ']
      if not int(year) in test_years:
        drops.append(i)
    test = test.drop(drops)
    return train, test
#selects specfic areas for where respondent is from for test and train data
def select_areas(train, test, train_areas, test_areas):
    #drops NaN values
    train = train.dropna(subset=['REGION OF RESIDENCE, AGE 16'])
    test = test.dropna(subset=['REGION OF RESIDENCE, AGE 16'])
    drops = []
    #goes through dataset and drops rows that aren't in train year area
    for i, row in train.iterrows():
        area = train.at[i,'REGION OF RESIDENCE, AGE 16']
        if int(area) not in train_areas:
            drops.append(i)
    train = train.drop(drops)
    drops = []
    #goes through dataset and drops rows that aren't in test year area
    for i, row in test.iterrows():
        area = test.at[i,'REGION OF RESIDENCE, AGE 16']
        if not int(area) in test_areas:
            drops.append(i)
    test = test.drop(drops)
    return train, test
#splits dataset into 80% as train and 20% as test sets. 
def normal_split(df):
    train, test = train_test_split(df, test_size=0.2)
    return train, test

'''
LOADING AND FORMATING THE DF
'''
#removes independents from dataset and takes in dataset as dataframe
def data_work():
    df = pd.read_csv("data.csv", low_memory=False)
    drops = []
    for i, row in df.iterrows():
        word = df.at[i,'POLITICAL PARTY AFFILIATION_labels']
        if "REP" not in str(word) and "DEM" not in str(word):
            drops.append(i)
    df = df.drop(drops)
    return(df)
'''
FUNCTIONS TO CHOOSE CERTAIN SETS OF COLUMNS
'''
def sample_sizes(allcols, sizes, n):
    cols = []
    for size in sizes:
        for j in range(n):
            temp = []
            for i in range(size):
                temp.append(allcols[random.randint(0,(len(allcols)-1))])
            cols.append(temp)
    return cols
#Processes through collumns and picks only valid collumns from txt file
'''
TRAIN AND TEST OPTIONS (uncomment 1 per test)
'''
def column_work():
    with open("qv1.txt") as f:
        precols = f.read().split("\n")
    cols = []
    for c in precols:
    	if not "label" in c:
    		cols.append(c)
    #specific test       
    return cols
'''
RUNNING THE NEURAL NET
'''
def run_network(allcols, dftrain, dftest, verbose=1):
    train_results = []
    test_results = []
    for col in allcols:
        print("Testing size:", len(col))

        #set collumns and removing null rows
        workercol = col.copy()
        workercol.append("POLITICAL PARTY AFFILIATION")
        train = dftrain[workercol]
        test = dftest[workercol]
        train = train.dropna()
        test = test.dropna()

        train = train.values.astype(float)
        test = test.values.astype(float)


        #normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(train)
        test = scaler.fit_transform(test)

        #setup inputs
        Xtrain = train[:,0:(len(col))]
        Ytrain = train[:,(len(col))]
        Xtest = test[:,0:(len(col))]
        Ytest = test[:,(len(col))]

        #one hot encode Y values
        Ytrain = to_categorical(Ytrain)
        Ytest = to_categorical(Ytest)

        #defining the model
        model = Sequential()

        #first layer, set hidden nodes, input demtions and, activation function 
        model.add(Dense(len(col)*2//3, input_dim=(len(col)), activation='relu'))

        #dropout
        model.add(Dropout(0.5))

        #layer with set number of inputs * 2/3 hidden nodes and relu activation function
        model.add(Dense(len(col)*2//3, activation='relu'))

        #dropout        
        model.add(Dropout(0.5))

        #final layer 2 output units
        model.add(Dense(2, activation='softmax'))

        #compiling model and setting evaluation metrics
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mse'])       

        #running the model with training set
        history = model.fit(Xtrain, Ytrain, epochs=30, batch_size=32,verbose=verbose)

        #running test set
        score = model.evaluate(Xtest, Ytest, batch_size=32)


        train_results.append(history.history)
        test_results.append(score)
        print(col, score)
    return train_results, test_results