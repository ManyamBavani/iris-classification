# Iris Flower Classification Using Machine Learning

## 📌 Project Overview

This project implements a Machine Learning classification system to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

The project uses two Machine Learning algorithms — Logistic Regression and Decision Tree — to train and predict flower species based on their measurements. It also compares model performance using evaluation metrics and visualizations.

## 🎯 Objectives

- Understand the Iris dataset.
- Perform Exploratory Data Analysis (EDA).
- Preprocess and standardize the dataset.
- Train and test Machine Learning models.
- Evaluate model performance using classification metrics.
- Visualize confusion matrices and decision boundaries using PCA.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── iris.py
├── iris.txt
├── README.md
└── requirements.txt
```

## 📊 Dataset

The project uses the Iris dataset stored in `iris.txt`.

The dataset contains flower measurements and a target column named `species`.

**Input features:**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

**Target variable:** `species`

The target contains three flower classes:
- Setosa
- Versicolor
- Virginica

## ⚙️ Project Workflow

1. **Load the Dataset:** Read the dataset using Pandas.
2. **Exploratory Data Analysis:** Visualize feature relationships using a pairplot.
3. **Train-Test Split:** Divide the dataset into 80% training data and 20% testing data.
4. **Feature Scaling:** Standardize features using StandardScaler.
5. **Model Training:** Train Logistic Regression and Decision Tree classifiers.
6. **Model Evaluation:** Calculate accuracy, precision, recall, F1-score, and classification reports.
7. **Confusion Matrix:** Visualize correct and incorrect predictions using heatmaps.
8. **PCA Visualization:** Reduce the feature dimensions to two components to visualize decision boundaries.

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression
A classification algorithm used to predict the species of an Iris flower based on its measurements.

### 2. Decision Tree Classifier
A tree-based classification algorithm that learns decision rules from the training data.

## 📈 Evaluation Metrics

The project evaluates both models using:

- **Accuracy:** Percentage of correct predictions.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures how many actual instances are correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications.

## 💻 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Iris-Flower-Classification.git
```

### Step 2: Open the Project

Open the project folder in Visual Studio Code.

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

Alternatively, install the libraries directly:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Step 4: Check the Dataset Path

Ensure that `iris.txt` is available in the project folder and update the dataset path in `iris.py` to:

```python
df = pd.read_csv("iris.txt")
```

### Step 5: Run the Project

```bash
python iris.py
```

The program displays dataset information, model evaluation results, confusion matrix heatmaps, and PCA decision boundary visualizations.

## 📦 Requirements

Create a file named `requirements.txt` containing:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## 🌟 Features

- Dataset loading and exploration
- Data visualization
- Feature standardization
- Multiple Machine Learning classifiers
- Model performance comparison
- Confusion matrix visualization
- PCA-based decision boundary visualization
