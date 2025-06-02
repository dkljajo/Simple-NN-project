# Simple-NN-project

# Binary Classification with Neural Networks in TensorFlow

This project demonstrates how to build a simple binary classification model using synthetic data, `pandas` for data manipulation, `scikit-learn` for preprocessing, and `TensorFlow` for neural network modeling.

## 🚀 Overview

The model predicts whether a user will make a purchase based on:
- `VisitDuration` (time spent on a website)
- `PagesVisited` (number of pages visited)

The target label `Purchase` is synthetically generated based on these features.

## 🧠 Model Architecture

- **Input Layer:** 2 features (`VisitDuration`, `PagesVisited`)
- **Hidden Layer:** 10 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation (for binary classification)

## 📦 Dependencies

Ensure you have the following libraries installed:

```bash
pip install numpy pandas scikit-learn tensorflow
📊 How It Works
Step 1: Data Generation
python
Copy
Edit
import numpy as np
import pandas as pd

# Synthetic features and labels
np.random.seed(0)
features = np.random.rand(200, 2)
labels = (features[:, 0] + features[:, 1] > 1).astype(int)

df = pd.DataFrame(features, columns=['VisitDuration', 'PagesVisited'])
df['Purchase'] = labels
Step 2: Preprocessing
python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[['VisitDuration', 'PagesVisited']],
    df['Purchase'],
    test_size=0.2,
    random_state=42
)
Step 3: Model Training
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10)
Step 4: Evaluation
python
Copy
Edit
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
📈 Results
After training, the model achieved ~80% accuracy on the test set:

yaml
Copy
Edit
Test Accuracy: 0.8000
Note: CUDA-related warnings are safe to ignore if you're running TensorFlow on CPU.

🛠 Future Improvements
Add early stopping and validation splits

Introduce more complex synthetic features or use real data

Test different activation functions and optimizers
