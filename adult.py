
# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.

import numpy as np
import pandas as pd


def read_csv_1(data_file):
    data_file = pd.read_csv(data_file)
    data_file.drop(columns = ['fnlwgt'], inplace = True)
    return data_file
    
"""
## 10 Points
- (a) the number of instances,
- (b) a list with the attribute names,
- (c) the number of missing values,
- (d) a list of the attribute names with at 1 least one missing value,
- (e) the percentage of instances corresponding to individuals whose education level is Bachelors or Masters (real number rounded to the first decimal digit).
"""

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    num_instances = len(df)
    return num_instances

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    attribute_list = list(df)
    return attribute_list

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    df = df.replace(r'^\s+$', np.nan, regex=True)
    # Number of NaN values
    return df.isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    return list(df.columns[df.isnull().any()])

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    bac = df.loc[df.education == 'Bachelors', 'education'].count()
    master = df.loc[df.education == 'Masters', 'education'].count()
    total = len(df['education'])

    percent = (bac + master) / total
    return round(percent, 3)

"""
## 10 points
### Drop all instances with missing values. Convert all attributes (except the class) to numeric using one-hot encoding. Name the new columns using attribute values from the original data set. Next, convert the class values to numeric with label encoding.
"""

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    df = df.dropna(axis=0, how='any')
    return df
    

# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.

def one_hot_encoding(df):
    encoding_df = data_frame_without_missing_values(df)
    # delete 'class' from the list
    encoding_df.drop(columns = ['class'], inplace = True)
    encoding_df_fianl = pd.get_dummies(encoding_df, prefix = column_names(encoding_df),
                                      columns = column_names(encoding_df))
    return encoding_df_fianl
    
from sklearn.preprocessing import LabelEncoder

# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding.

def label_encoding(df):
    df = data_frame_without_missing_values(df)
    label_encoder = LabelEncoder()
    class_encode = label_encoder.fit_transform(df['class'])
    return pd.Series(class_encode)

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values.


"""
## 10 points
### Build a decision tree and classify each instance to one of the <= 50K and > 50K categories. Compute the training error rate of the resulting tree.
"""

from sklearn.tree import DecisionTreeClassifier

# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train.
# Return a pandas series with the predicted values.

def dt_predict(X_train,y_train):
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train,y_train)
    #prediction
    y_pred = classifier.predict(X_train)
    return pd.Series(y_pred)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.

from sklearn.metrics import confusion_matrix

def dt_error_rate(y_pred, y_true):
    matrix = confusion_matrix(y_true, y_pred)
    # outcome values order in sklearn
    tp, fn, fp, tn = confusion_matrix(y_true,y_pred).reshape(-1)
    error_rate = (fp + fn) / (tp + fn + fp + tn)
    return round(error_rate,3)
