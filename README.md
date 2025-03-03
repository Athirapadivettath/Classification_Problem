# Classification_Problem

## Objective
This project applies various classification algorithms to the Breast Cancer dataset to predict whether a tumor is malignant or benign. The goal is to evaluate and compare model performance.

## Dataset
- The dataset used is the **Breast Cancer dataset**, loaded using `load_breast_cancer` from `sklearn.datasets`.
- It contains multiple features describing tumor characteristics and a target variable indicating malignancy.

## Implementation Steps
1. **Loading and Preprocessing**
   - Load the dataset into a Pandas DataFrame.
   - Perform feature scaling using `StandardScaler`.
   - Split the data into training and testing sets.

2. **Classification Models**
   - **Logistic Regression**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Support Vector Classifier (SVC)**
   - **k-Nearest Neighbors (k-NN)**

3. **Evaluation Metrics**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-score)

4. **Results & Visualization**
   - Compare accuracy scores for all models.
   - Visualize model performance using a bar chart.

## Results
- The **best-performing model** is **Logistic Regression** with an accuracy of **97.37%**.
- The **worst-performing model** is **Decision Tree** with an accuracy of **94.74%**.
- The model performance is visualized in a bar chart.

## Repository Structure
- `classification_assignment.ipynb` - Jupyter Notebook containing the implementation.
- `README.md` - This file describing the project.


## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

