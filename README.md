# mhmdn1-490_PS3_Titanic


---

# Titanic Neural Network Project

This project implements a feed-forward neural network to predict passenger survival on the Titanic based on various features. The model is built using TensorFlow and follows a supervised learning approach with preprocessed data.

## Project Structure
```
/Titanic-NeuralNetwork
│
├── README.md               # Documentation
├── neural_network.py       # Python script with NeuralNetwork class and BuildDataset function
├── run_model.ipynb         # Jupyter notebook for demonstration
└── data.csv                # Titanic dataset
```

## Overview

- **`neural_network.py`**: Contains a `BuildDataset` function for data preprocessing and a `NeuralNetwork` class for constructing, training, and evaluating the model.
- **`run_model.ipynb`**: Jupyter notebook to demonstrate the use of the `BuildDataset` function and `NeuralNetwork` class for training and evaluating the model, along with sample predictions.
- **`data.csv`**: The dataset used for training and testing the model.

## Installation

Ensure you have Python and the following libraries installed:
- TensorFlow
- scikit-learn
- pandas
- NumPy

You can install the required libraries using:
```bash
pip install tensorflow scikit-learn pandas numpy
```

## Usage

1. **Run the Python Script**: Execute `neural_network.py` to preprocess the data and train the model.
2. **Jupyter Notebook**: Open and run `run_model.ipynb` for a step-by-step guide on data preprocessing, training, evaluation, and sample predictions.

## How It Works

1. **Data Preprocessing**:
   - Missing values in `Age` and `Fare` are filled with their medians.
   - Categorical features (`Sex` and `Embarked`) are label-encoded.
   - The data is split into training and testing sets and scaled.

2. **Model Architecture**:
   - The neural network has three layers:
     - Input layer with 64 units (ReLU activation)
     - Hidden layer with 32 units (ReLU activation)
     - Output layer with 1 unit (sigmoid activation)

3. **Training**:
   - The model is compiled with `Adam` optimizer and `binary_crossentropy` loss.
   - It trains for 50 epochs by default, with an 80-20 training-validation split.

4. **Evaluation**:
   - Model performance is measured using accuracy and loss metrics.
   - The notebook displays predictions versus actual outcomes for better insight.
