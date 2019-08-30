#!/usr/bin/env python
# coding: utf-8

# ## Data science Assessment: Problem, Analysis and Solution

# <b>Title : Walmart - Store Sales Forecasting</b>
# 
# Problem Statement : You are provided with historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments, and you are tasked with predicting the department-wide sales for each store.
# 
# In addition, Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. Part of the challenge presented by this competition is modeling the effects of markdowns on these holiday weeks in the absence of complete/ideal historical data.
# 
# More details regarding the problem can be find from kaggle.com: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data.

# ## What's included in the Solution

# 1. A detailed analysis and solution file with descriptions
# 2. A scoring metric to decide which model/algorithm is "better."
# 3. Feature selection framework to implement an optimized model selection approach.

# ## Analysis and Solution

# In[4]:


# list of all the libraries required for the solution
import pandas as pd
# linear algebra
import numpy as np
# Seaborn library for plots
import seaborn as sns; sns.set(style="ticks", color_codes=True)
# mlflow for workflow management
import mlflow
from mlflow import tracking
# Derived Model functions
from helperFunctions import dataPreparation, modelSelection, trainModel, fnFeatureImportance, scatter
import settings

def main():
    # Step 1 : Data Acqusition & Loading
    # 1.1 Load the traing Data Set
    trainingData = pd.read_csv(settings.TRAINING_DATA)
    # 1.2 Load the test Data Set
    testData = pd.read_csv(settings.TESTING_DATA)
    # 1.3 Load the test features Data Set
    features = pd.read_csv(settings.FEATURES_DATA)
    # 1.4 Load the test Stores Data Set
    stores = pd.read_csv(settings.STORES_DATA)
    
    # Step 2 : Data Integration based on project requirements
    # 2.1 Merge all the train, features and stores data to build a combined training set
    trainingData = trainingData.merge(stores, how='left').merge(features, how='left')
    # 2.2 Merge all the testData, features and stores data to build a combined test data set
    testData = testData.merge(stores, how='left').merge(features, how='left')
    # 2.3 Save the data as artifact
    mlflow.log_artifact(trainingData)
    mlflow.log_artifact(testData)
    
    # Step 3 : Data Analysis
    # 3.1 Use Scatter plots to visualize the data and it's landscape
    scatter(trainingData, 'Fuel_Price')
    scatter(trainingData, 'Size')
    scatter(trainingData, 'CPI')
    scatter(trainingData, 'Type')
    scatter(trainingData, 'IsHoliday')
    scatter(trainingData, 'Unemployment')
    scatter(trainingData, 'Temperature')
    scatter(trainingData, 'Store')
    scatter(trainingData, 'Dept')
    
    # 3.2 Use pair plot to Visualize the pairs
    sns.pairplot(trainingData, vars=['Weekly_Sales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'])
    # 3.2 Use pair plot to Visualize the pairs Continuation...
    sns.pairplot(trainingData.fillna(0), vars=['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
    
    # Step 4 : Data Preparation & Manipulations
    # 4.1 Preparation of the training Dataset
    trainingData = dataPreparation(trainingData)
    # 4.2 Preparation of the test Dataset
    testData = testData.fillna(0)
    column_date = testData['Date']
    # Assign the testData values to X_test variable
    X_test = dataPreparation(testData)
    
    # Step 5 : Model Training and Cross Validation to find the Best Model from the model selection list
    best_model, selectedModel, X_train, y_train = modelSelection(trainingData)
    print("The Best model is : ")
    print(best_model)
    # log some parameters
    mlflow.log_param("Best Model", best_model)
    mlflow.log_param("Selected Model", selectedModel)
    
    # Step 6 : Model Optimization
    # Apply model based feature importance selection if the selected model is not neuralnet
    if (selectedModel != settings.NEURALNET):
        # feature importance    
        reducedSet = fnFeatureImportance(best_model, X_train)    
        # reduced subset with important features
        X_train = X_train[reducedSet].reset_index(drop = True)        
        # Assign test Data to get the reduced Data Set
        X_test = X_test[reducedSet].reset_index(drop = True)
        # Retrain the model with the reduced set
        best_model = trainModel(selectedModel, X_train, y_train)
        
    # Step 7 : Final Model prediction based on the best model selection
    # Execute the prediction Algorithm
    predicted_test = best_model.predict(X_test)
    
    # Step 8 : Output Data Preparation
    # Integrate the Prediction Results with the input(Test) data sheet
    testData['Predicted_WeeklySales'] = predicted_test
    testData['Prediction_Date'] = column_date
    testData['Prediction_Id'] = testData['Store'].astype(str) + '_' +  testData['Prediction_Date'].astype(str)
    testData = testData[['Prediction_Id', 'Store', 'Prediction_Date', 'Predicted_WeeklySales']]
    # Produce the Output Result file
    testData.to_csv(settings.OUTPUT_FILE, index=False)
    # log the Model Output as an artifact
    mlflow.log_artifact(testData)
    
#Main function for the program execution
if __name__ == '__main__':
    # Call main function to execute the application logic
    main()

