# Titanic Dataset Analysis

# TASK 01

1. **Importing libraries:**
    - numpy, pandas, matplotlib, seaborn, sklearn.preprocessing
2. **Loading the dataset:**
    - load the dataset "Titatnic_Dataset.csv" using pd.read_csv()
3. **Initial Exploration**
    - Displayed the first few rows using head().
        Used info() and isnull().sum() to inspect data types and missing values.

4.  **Handling Missing Values**
    - Filled missing values in:
        Age with the mean.
        Embarked with the mode.
    - Dropped the Cabin column due to too many missing values.

5. **Feature Selection**
    - Dropped Name and Ticket columns as they are not useful for modeling.

6.  **Encoding Categorical Variables**
    - Used label encoding for Sex: 'male' → 0, 'female' → 1.
    - Applied one-hot encoding to the Embarked column (with drop_first=True).

7.  **Feature Scaling**
    - Standardized numerical columns: Age, Fare, SibSp, Parch using StandardScaler.

8.  **Outlier Detection and Removal**
    - Plotted boxplots for the numeric columns.
    - Removed outliers using the IQR method for the selected numeric features.
    - Plotted boxplots again to confirm outlier removal.

# TASK-02

1.  **Summary Statistics**
    - Used describe(include="all") to calculate mean, median, standard deviation, min, max, etc.

2.  **Data Visualization**
    - Histograms of numeric features to understand their distributions.
    - Boxplots to visualize data spread and detect outliers.
    - Pairplot to see pairwise relationships colored by the Survived column.
    - Correlation Matrix with heatmap to evaluate relationships between features.

**Output**
    - Cleaned, transformed dataset ready for machine learning.
    - Visual insights into feature distributions, correlations, and survival trends.

# TASK-03

In this task we have used "Housing.csv" Dataset for House Price Prediction.

1.  **Importated libraries** 
2.  **Converted Categorical Data to Numerical**
2.  **Split the Data into Train and Test Sets**
3.  **Used Simple Linear Regression Model to train and test** 
4.  **Evaluated the models accuracy by calculating:**
    - Mean Squared Error
    - Mean Absolute Error
    - R-Squared Value
5.  **Plotted the Predicted VS Actual Price of the houses to interpret visually How well the model Predicted and calculated the Model's Coefficient.**


# TASK-04

Logistic Regression on Breast Cancer Dataset
This project demonstrates a complete binary classification pipeline using Logistic Regression on the Breast Cancer Wisconsin dataset.

1.  **Data Preprocessing:**
    - Dropped irrelevant columns (id, Unnamed: 32)
    - Encoded target: M → 1 (Malignant), B → 0 (Benign)

2.  **Model Pipeline:**
    - Train/test split (80/20)
    - Feature standardization using StandardScaler
    - Model training using LogisticRegression
3.  **Model Evaluation:**
    - Confusion Matrix
    - Precision and Recall
    - ROC-AUC Score and ROC Curve
    - Threshold tuning at 0.3 and 0.5
4.  **Concept Explanation:**
    - Sigmoid function and how logistic regression outputs probabilities

# TASK-05

Heart Disease Classification using Decision Trees and Random Forest.
This project applies machine learning models to predict the presence of heart disease using the **UCI Heart Disease Dataset**.

1. **Train a Decision Tree Classifier and Visualize the Tree**  
   - Used `sklearn.tree.DecisionTreeClassifier`  
   - Visualized using `plot_tree()`

2. **Analyze Overfitting and Control Tree Depth**  
   - Compared default vs pruned (`max_depth=4`) trees  
   - Evaluated accuracy and overfitting impact

3. **Train a Random Forest and Compare Accuracy**  
   - Used `RandomForestClassifier`  
   - Compared with pruned Decision Tree accuracy

4. **Interpret Feature Importances**  
   - Plotted top contributing features from the Random Forest model

5. **Evaluate Using Cross-Validation**  
   - Performed 5-fold cross-validation  
   - Achieved ~99.7% average accuracy

# TASK-06

This project applies K-Nearest Neighbors (KNN) to the Iris dataset using Scikit-learn. The goal is to explore model performance with varying parameters and visualize results.

1. **Data Loading & Normalization**  
   - Loaded `Iris.csv` and normalized features using `StandardScaler` for better model performance.

2. **Model Setup**  
   - Used `KNeighborsClassifier` from `sklearn` for classification.

3. **K Value Experimentation**  
   - Tested different values of K (e.g., 3, 5, 7) to observe changes in model accuracy.

4. **Model Evaluation**  
   - Evaluated performance using `accuracy_score` and visualized predictions using a confusion matrix.

5. **Decision Boundary Visualization**  
   - Applied PCA to reduce data to 2D and plotted decision boundaries for different K values.

