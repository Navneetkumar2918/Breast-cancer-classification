# Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project focuses on building a Machine Learning model to classify breast tumors as Malignant or Benign using medical diagnostic data. It uses Python and the Scikit-learn library to load, preprocess, train, evaluate, and test the model. The main goal is to support early cancer detection by providing accurate predictions.

## 📊 Dataset

The dataset is loaded from Scikit-learn’s built-in breast cancer dataset. It contains various medical features such as radius, texture, perimeter, area, smoothness, and more.

```python
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```

Each record represents one patient sample with multiple attributes and a target label.

* 0 → Malignant (Cancerous)
* 1 → Benign (Non-cancerous)

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Google Colab / Jupyter Notebook

## 📂 Data Preparation

The dataset is converted into a Pandas DataFrame for easier analysis.

Target labels are added using:

```python
data_frame['lable'] = breast_cancer_dataset.target
```

Data analysis is performed using:

* data_frame.head()
* data_frame.tail()
* data_frame.info()
* data_frame.describe()
* data_frame.isnull().sum()
* value_counts()

These steps help in understanding the dataset and checking for missing values.

## 🔄 Data Splitting

The dataset is divided into training and testing sets using:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```

80% data is used for training and 20% for testing.

## 🤖 Model Training

A Logistic Regression model is used for classification.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

The model learns patterns from the training data.

## 📈 Model Evaluation

Model accuracy is calculated for both training and testing data.

```python
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
```

This helps in checking the performance and reliability of the model.

## 🔍 Prediction System

Custom input data is provided to predict new cases.

```python
input_data = (...)
```

The input is converted into NumPy array and reshaped before prediction.

The output is displayed as:

* The Breast Cancer is Malignant
* The Breast Cancer is Benign

## ▶️ How to Run the Project

1. Open the project in Google Colab or Jupyter Notebook.
2. Install required libraries.
3. Run all cells step by step.
4. Train the model.
5. Provide input data for prediction.

## 📌 Applications

* Medical diagnosis support
* Early cancer detection
* Learning ML classification techniques
* Academic projects

## 👨‍💻 Author

Navneet Kumar

## ⚠️ Disclaimer

This project is created for educational purposes only and should not be used as a replacement for professional medical diagnosis.
