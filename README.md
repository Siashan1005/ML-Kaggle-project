# ML-Kaggle-project
# Project Overview
This project was part of a Kaggle competition for CompSci 671, where the goal was to develop a predictive model for real estate pricing using structured tabular data. The dataset contained various attributes related to property listings, including location, host details, property type, and review metrics. The objective was to apply exploratory data analysis (EDA), feature engineering, and machine learning models to optimize predictive performance.

# Exploratory Data Analysis (EDA) and Data Visualization
I began with a dataset overview using df.info() and calculated the proportion of missing values to identify potential preprocessing steps. A heatmap was generated to visualize missing data distribution, and columns with over 90% missing values were removed. I handled missing values by applying mode imputation for categorical features and median imputation for numerical features to retain data integrity.

For feature engineering, I transformed date-related variables, converting host_since, first_review, and last_review into numerical features representing the number of years since the event occurred. Categorical variables such as room_type and host_response_time were one-hot encoded, while neighborhood scores were mapped numerically to enhance interpretability. Histograms, box plots, and scatter plots were used to analyze distributions and detect outliers.

A Random Forest model was used for feature importance analysis, revealing that room_type, minimum_nights, longitude, and host_listings_count were the most influential factors in predicting property prices. These insights guided the selection of relevant features for model training.

# Model Selection and Training
I experimented with two models: XGBoost and CatBoost, both gradient boosting algorithms known for their efficiency with structured data.

XGBoost: Selected for its strong predictive power and built-in missing value handling. Hyperparameter tuning (learning rate, max depth, number of estimators) was performed using grid search with 5-fold cross-validation.
CatBoost: Chosen for its ability to handle categorical features natively without extensive preprocessing. It was optimized by tuning depth and learning rate, leveraging its ordered boosting mechanism.
For model evaluation, I used Root Mean Squared Error (RMSE) and R² scores, validating performance on a train-validation split (80-20%). CatBoost outperformed XGBoost, likely due to its categorical feature handling and reduced overfitting.

# Final Submission and Reflection
The final model predictions were saved and submitted to Kaggle, where performance was assessed against competitors. Challenges included handling missing values, balancing feature selection with model complexity, and optimizing hyperparameters. This project reinforced key machine learning concepts, particularly feature engineering, model interpretability, and cross-validation in structured datasets.
