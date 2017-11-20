#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:22:25 2017

@author: vzabulskyy
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter

# Importing the dataset
dataset = pd.read_csv("../Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
'''
# Handle missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', axis=0)
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
'''
# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_test)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
