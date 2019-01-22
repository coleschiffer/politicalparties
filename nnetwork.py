import numpy as np
import pandas as pd
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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
FUNCTIONS TO CHOOSE CERTAIN SETS OF COLUMNS
'''
# function to generate all the sub lists
def sub_lists(list1):
    # store all the sublists
    sublist = [[]]
    for i in range(len(list1) + 1):
        for j in range(i + 1, len(list1) + 1):
            sub = list1[i:j]
            sublist.append(sub)
    cols = [x for x in sublist if x != []]
    return cols

# function to generate all sub list of sizes with multiples of 5
def sample_sizes(allcols, sizes, n):
    cols = []
    for size in sizes:
        for val in range(n):
            temp = []
            temp_cols = random.sample(range(len(allcols)), size)
            for num in temp_cols:
                temp.append(allcols[num])
            cols.append(temp)
    return cols
'''
LOADING AND FORMATING THE DF
'''
#takes in dataset as dataframe and bianarily categorizes pollitical beliefs 
def data_work():
    df = pd.read_csv("data.csv", low_memory=False)
    drops = []
    for i, row in df.iterrows():
      word = df.at[i,'POLITICAL PARTY AFFILIATION_labels']
      if "REP" in str(word):
        df.at[i,'POLITICAL PARTY AFFILIATION_labels'] = 0
      elif "DEM" in str(word):
        df.at[i,'POLITICAL PARTY AFFILIATION_labels'] = 1
      else:
      	drops.append(i)
    df = df.drop(drops)
    return(df)

#Processes through collumns and picks only valid collumns from txt file
def column_work():
    with open("qv1.txt") as f:
        precols = f.read().split("\n")
    cols = []
    for c in precols:
    	if not "label" in c:
    		cols.append(c)
    return cols

'''
RUNNING THE NEURAL NET
'''
def run_network(allcols, dftrain, dftest, verbose=1):
    
    train_results = []
    test_results = []
    for col in allcols:
        print("Testing size:", len(col))
        print("Testing:", col)

        workercol = col.copy()
        workercol.append("POLITICAL PARTY AFFILIATION_labels")

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

        #defining the model
        model = Sequential()
        
        #first layer, set hidden nodes, input demtions and, activation function 
        model.add(Dense(len(workercol)*2//3, input_dim=(len(col)), activation='relu'))

        #dropout
        model.add(Dropout(0.5))

        #layer with set number of inputs * 2/3 hidden nodes and relu activation function
        model.add(Dense(len(workercol)*2//3, activation='relu'))

        #dropout
        model.add(Dropout(0.5))

        #final layer with 1 binary output 
        model.add(Dense(1, activation='sigmoid'))

        #compiling model and setting evaluation metrics
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mse'])

        #running the model on the training set
        history = model.fit(Xtrain, Ytrain, epochs=30, batch_size=32, verbose=verbose)

        train_results.append(history.history)
        test_results.append(score)

    return train_results, test_results
