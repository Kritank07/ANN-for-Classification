# ANN for Classification using PyTorch

A multi-class classification project built using an Artificial Neural Network (ANN) in PyTorch, trained on the **Date Fruit Dataset**. The project includes two approaches — one using all original features and one using **PCA-reduced features** — to compare model performance and efficiency.

---

## 📌 Problem Statement

Classify different varieties of date fruits based on their physical and morphological features using a fully connected neural network.

---

## 📂 Dataset

- **Dataset:** Date Fruit Dataset (`DateFruit_Dataset.csv`)
- **Target Column:** `Class` (7 unique fruit varieties)
- **Features:** Numeric physical attributes of date fruits

---

## 🏗️ Project Workflow
Raw Data → Preprocessing → Model Training → Evaluation

↓

(Also) PCA Reduction → Model Training → Evaluation

### Steps:
1. Load and explore the dataset
2. Encode target labels using `LabelEncoder`
3. Split data into train/test sets (80/20)
4. Standardize features using `StandardScaler`
5. Convert data to PyTorch tensors and create `TensorDataset` and `DataLoader`
6. Build and train ANN model
7. Evaluate model accuracy
8. Apply PCA (4 components) and repeat training & evaluation

---

## 🧠 Model Architecture

### Approach 1 — Full Features
Input Layer  →  Linear(n_features, 64)  →  ReLU

Hidden Layer →  Linear(64, 64)          →  ReLU

Output Layer →  Linear(64, 7)

### Approach 2 — PCA Reduced (4 components)
Input Layer  →  Linear(4, 32)  →  ReLU

Hidden Layer →  Linear(32, 32) →  ReLU

Output Layer →  Linear(32, 7)

---

## ⚙️ Training Configuration

| Parameter       | Value              |
|-----------------|--------------------|
| Loss Function   | CrossEntropyLoss   |
| Optimizer       | Adam               |
| Epochs          | 100                |
| Batch Size      | 32                 |

---

## 📊 Evaluation

The model is evaluated on the test set using:
- **Accuracy** = (Correct Predictions / Total Samples) × 100

---

## 🛠️ Tech Stack

| Tool         | Purpose                        |
|--------------|--------------------------------|
| Python       | Core language                  |
| PyTorch      | Neural network & training      |
| Scikit-learn | Preprocessing, PCA, splitting  |
| Pandas       | Data loading & manipulation    |
| NumPy        | Numerical operations           |
