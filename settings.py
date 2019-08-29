#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Base libraries
import pandas as pd

# Input Data files settings
TRAINING_DATA = 'train.csv'
TESTING_DATA = 'test.csv'
FEATURES_DATA = 'features.csv'
STORES_DATA = 'stores.csv'

# Hand picked models to train
NEURALNET = "neuralnet"
EXTRATREESREGRESSOR = "extraTreesRegressor"
RANDOMFORESTREGRESSOR = "randomForestRegressor"

# Model Array contains the list of models to be evaluated
MODELARRAY = [NEURALNET, EXTRATREESREGRESSOR, RANDOMFORESTREGRESSOR]

#Write OutPut File to the Output location
OUTPUT_FILE = 'Output.csv'

