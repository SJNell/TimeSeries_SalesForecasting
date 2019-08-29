#!/usr/bin/env python
# coding: utf-8

# In[1]:


# List of all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import all the required Model libraries
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
# Import Model Variables
import settings

def fnFeatureImportance(fit_model, x_train):
    """
    Model based feature importance

    Parameters
    ----------
    fit_model : Decision Tree Classifier Trained model
    x_train: pandas dataframe Training data

    Returns
    -------
    list Important features based on the model.
    """

    #get important features
    importances = fit_model.feature_importances_

    important_feature_index = [i for i, x in enumerate(importances) if x != 0]
    important_features_to_model = x_train.iloc[:, important_feature_index]

    #save features with importance > 0
    reducedSet = important_features_to_model.columns 

    return reducedSet;

def scatter(trainingData, column):
    plt.figure()
    plt.scatter(trainingData[column] , trainingData['Weekly_Sales'])
    plt.ylabel('Weekly_Sales')
    plt.xlabel(column)

# Define the Suitable Models for Analysis
def extraTreesRegressor(train_x, train_y):
    model = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)
    model.fit(train_x, train_y)
    return model

def randomForestRegressor(train_x, train_y):
    model = RandomForestRegressor(n_estimators=100,max_features='log2', verbose=1)
    model.fit(train_x, train_y)
    return model

def neuralnet(train_x, train_y):
    model = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', verbose=3)
    model.fit(train_x, train_y)
    return model

# Predict function to execute the Model predictions
def predict_(m, test_x):
    return pd.Series(m.predict(test_x))

# Train Model to execute the Model
def trainModel(model, train_x, train_y):
    if (model == settings.EXTRATREESREGRESSOR):
        return extraTreesRegressor(train_x, train_y)
    elif (model == settings.RANDOMFORESTREGRESSOR):
        return randomForestRegressor(train_x, train_y)
    elif (model == settings.NEURALNET):
        return neuralnet(train_x, train_y)

# Function for Error Calculations
def calculate_error(test_y, predicted, weights):
    return mean_absolute_error(test_y, predicted, sample_weight=weights)

# Model selection after training and nested Cross Validation
def modelSelection(trainingData):
    # 5.1 Generating Data  splits for Nested Cross Validation
    tscv = TimeSeriesSplit(n_splits = 4)
    # Intialize the function variables
    modelArray = settings.MODELARRAY
    best_model = None
    best_error = np.iinfo(np.int32).max
    selctedModel = None
    # Executing the logic to calculate the error factor and to pick the right model
    for model in modelArray:
        for train_index, test_index in tscv.split(trainingData.groupby(["Store", "Dept"])):
            # Generate cross Validation training and test data
            cv_train, cv_test = trainingData.iloc[train_index], trainingData.iloc[test_index]
            # Define the model training and test variables
            y_train = cv_train['Weekly_Sales']
            X_train = cv_train.drop(columns=['Weekly_Sales'])        
            y_test = cv_test['Weekly_Sales']
            X_test = cv_test.drop(columns=['Weekly_Sales'])       
            # Call trainModel function to training the selected models
            fit_model = trainModel(model, X_train, y_train)
            # Predict the values based on the fitted model
            predicted = predict_(fit_model, X_test)
            # Applying weightage to if is IsHoliday is true
            weights = X_test['IsHoliday'].replace(True, 5).replace(False, 1)
            # Calculating the error factor
            error = calculate_error(y_test, predicted, weights)
            # Logic to find the best model
            if error < best_error:
                # print('Find best model')
                best_error = error
                best_model = fit_model
                selctedModel = model
    return best_model, selctedModel, X_train, y_train

# Data Preparation and Cleansing
def dataPreparation(Data):
    # get dummy data for the Type field
    Data = pd.get_dummies(Data, columns=["Type"])
    Data[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = Data[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
    # Generate an additional field called Sales Month
    Data['Sales_Month'] = pd.to_datetime(Data['Date']).dt.month
    # Drop the Date field, Other insignificant data fields will be deleted on the basis of feature importance
    Data = Data.drop(columns=["Date"])
    return Data

