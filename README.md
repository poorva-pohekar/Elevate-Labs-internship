# Titanic Dataset Analysis

TASK 01

1. # Importing libraries:
    - numpy, pandas, matplotlib, seaborn, sklearn.preprocessing
2. # Loading the dataset:
    - load the dataset "Titatnic_Dataset.csv" using pd.read_csv()
3. # Initial Exploration
    - Displayed the first few rows using head().
        Used info() and isnull().sum() to inspect data types and missing values.

4.  # Handling Missing Values
    - Filled missing values in:
        Age with the mean.
        Embarked with the mode.
    - Dropped the Cabin column due to too many missing values.

5. # Feature Selection
    - Dropped Name and Ticket columns as they are not useful for modeling.

6.  # Encoding Categorical Variables
    - Used label encoding for Sex: 'male' → 0, 'female' → 1.
    - Applied one-hot encoding to the Embarked column (with drop_first=True).

7.  # Feature Scaling
    - Standardized numerical columns: Age, Fare, SibSp, Parch using StandardScaler.

8.  # Outlier Detection and Removal
    - Plotted boxplots for the numeric columns.
    - Removed outliers using the IQR method for the selected numeric features.
    - Plotted boxplots again to confirm outlier removal.

TASK-02

1.  # Summary Statistics
    - Used describe(include="all") to calculate mean, median, standard deviation, min, max, etc.

2.  # Data Visualization
    - Histograms of numeric features to understand their distributions.
    - Boxplots to visualize data spread and detect outliers.
    - Pairplot to see pairwise relationships colored by the Survived column.
    - Correlation Matrix with heatmap to evaluate relationships between features.

## Output
    - Cleaned, transformed dataset ready for machine learning.
    - Visual insights into feature distributions, correlations, and survival trends.