# Wine Quality Prediction using Artificial Neural Networks (ANN)

## Project Overview
This project aims to predict the quality of wine based on its physicochemical properties using an Artificial Neural Network (ANN). The dataset contains various chemical analyses of red wines, and the goal is to classify wine quality into discrete categories.

## Data

### Data Source
The dataset used in this project is `wine.csv`. It contains features related to the physicochemical properties of wine and a 'quality' score.

### Data Preprocessing
1.  **Feature Dropping**: The 'Id' column was dropped as it's an identifier and not relevant for prediction.
2.  **Target Variable Mapping**: The 'quality' column, representing the target variable, was mapped to numerical categories from 0 to 5 to align with the `sparse_categorical_crossentropy` loss function. The original quality values were `[3, 4, 5, 6, 7, 8]`.
3.  **Train-Test Split**: The dataset was split into training and testing sets with a `test_size` of 20% and `random_state` set to 42 for reproducibility.
4.  **Feature Scaling**: Features were scaled using `StandardScaler` to normalize their ranges. This is crucial for neural networks to ensure stable and efficient training.

## Methodology

### Model Architecture
An Artificial Neural Network (ANN) was constructed using TensorFlow/Keras with the following layers:
*   **Input Layer**: A `Dense` layer with 32 units, `relu` activation, and an `input_shape` of `(11,)` corresponding to the number of features.
*   **Hidden Layer**: A `Dense` layer with 16 units and `relu` activation.
*   **Output Layer**: A `Dense` layer with 6 units (matching the 6 unique quality categories) and `softmax` activation for multi-class classification.

```python
model = Sequential([
    Dense(32, activation = 'relu', input_shape=(11,)),
    Dense(16, activation = 'relu'),
    Dense(6, activation = 'softmax')
])
```

### Model Compilation
The model was compiled with the following parameters:
*   **Optimizer**: `adam`
*   **Loss Function**: `sparse_categorical_crossentropy` (suitable for integer-encoded labels)
*   **Metrics**: `accuracy`

```python
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
```

### Model Training
The model was trained for 100 epochs with a `batch_size` of 32, using the training data (`X_train`, `y_train`) and validated on the test data (`X_test`, `y_test`).

```python
history = model.fit(
    X_train, y_train,
    epochs = 100,
    batch_size = 32,
    validation_data = (X_test, y_test)
)
```

## Results

After training, the model achieved the following performance on the test set:
*   **Final Test Loss**: 0.9487
*   **Final Test Accuracy**: 64.63%

Training and validation accuracy/loss curves were plotted to visualize the model's learning progression over epochs.

## Model Saving
The trained model has been saved in the native Keras format for future use and deployment:

`my_wine_model.keras`

## How to Reproduce
To reproduce this project, follow these steps:
1.  Ensure you have `tensorflow`, `keras`, `pandas`, `scikit-learn`, and `matplotlib` installed.
2.  Load the `wine.csv` dataset.
3.  Execute the preprocessing steps outlined in the 'Data Preprocessing' section.
4.  Construct and compile the ANN model as specified in 'Model Architecture' and 'Model Compilation'.
5.  Train the model using the parameters from 'Model Training'.
6.  Evaluate the model and visualize the results.
