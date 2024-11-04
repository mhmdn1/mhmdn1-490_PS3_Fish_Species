from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf
import pandas as pd

# Function to build and prepare the dataset
def BuildDataset(data):
    # Fill missing 'Age' and 'Fare' with their respective medians
    data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())
    data.loc[:, 'Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    data.loc[:, 'Sex'] = le_sex.fit_transform(data['Sex'])

    le_embarked = LabelEncoder()
    data.loc[:, 'Embarked'] = data['Embarked'].fillna('S')  # Fill missing 'Embarked' with the most common value
    data.loc[:, 'Embarked'] = le_embarked.fit_transform(data['Embarked'])
    
    # Drop unused columns
    features = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1)
    target = data['Survived']
    
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Class to build and train the neural network
class NeuralNetwork:
    def __init__(self, input_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

# Example usage for your Jupyter notebook
if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    X_train, X_test, y_train, y_test = BuildDataset(data)
    nn = NeuralNetwork(input_shape=X_train.shape[1])
    nn.train(X_train, y_train, epochs=50)
    print("Evaluation:", nn.evaluate(X_test, y_test))

