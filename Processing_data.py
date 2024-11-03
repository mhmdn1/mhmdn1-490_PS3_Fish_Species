import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def prepare_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Convert 'Wage' from string to numeric by removing commas and converting to integer
    data['Wage'] = data['Wage'].str.replace(',', '').astype(int)
    
    # Select the first 600 entries
    data = data.iloc[:600]
    
    # Identify categorical columns
    categorical_columns = ['Club', 'League', 'Nation', 'Position']
    
    # One-hot encode the categorical data
    data = pd.get_dummies(data, columns=categorical_columns)
    
    # Split the data into features and target
    X = data.drop('Wage', axis=1)
    Y = data['Wage']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    return X_train, X_test, y_train, y_test

