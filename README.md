# Lung Cancer Prediction Project

This project aims to predict the likelihood of lung cancer based on various demographic and health-related factors. The dataset used for this project contains information on individuals' age, gender, smoking habits, and various symptoms associated with lung cancer.

## Files

-   `survey_lung_cancer.csv`: Dataset containing information about individuals and lung cancer status.
-   `Project.ipynb`: The code that outlines the entire project pipeline, including data loading, preprocessing, model building, evaluation, and visualization of results.
-   `README.md`: Documentation providing an overview of the project, usage instructions, and dependencies.

## Dependencies

-   NumPy
-   pandas
-   Matplotlib
-   scikit-learn

## Usage

1.  **Data Loading**: The project starts by loading the dataset `survey_lung_cancer.csv` using pandas.

2.  **Data Exploration**: Initial exploration of the dataset is performed to understand its structure, dimensions, and data types. This step helps in identifying any missing values or inconsistencies in the data.

3.  **Data Preprocessing**: The dataset undergoes preprocessing steps such as encoding categorical variables and handling missing values.

4.  **Model Building**:

    -   **Logistic Regression**: A logistic regression model is trained to predict lung cancer.
    -   **LASSO Regression**: LASSO regression, a type of linear regression with L1 regularization, is employed to identify significant features and predict lung cancer.
    -   **Random Forests**: A random forests model is trained to capture non-linear relationships and interactions among features for lung cancer prediction.
    -   **Support Vector Machines (SVM)**: SVMs is trained on the dataset to predict lung cancer.

5.  **Model Evaluation**: Each model's performance is evaluated using root mean squared logarithmic error (RMSLE) and accuracy score.

6.  **Feature Importance Visualization**: Feature importance is visualized for models to understand which features contribute most to the prediction.

## Results

The lung cancer prediction project yielded promising results across multiple models:

-   **Logistic Regression**: Achieved an RMSLE of 0.337 and accuracy score of 0.968 on the test set. Identified key predictors of lung cancer, including FATIGUE, SWALLOWING DIFFICULTY, CHRONIC DISEASE, ALLERGY, ALCOHOL CONSUMING.

-   **LASSO Regression**: Achieved an RMSLE of 0.309 and accuracy score of 0.968 on the test set. The model highlighted ALCOHOL CONSUMING, FATIGUE, ALLERGY, YELLOW FINGERS, COUGHING.

-   **Random Forests**: The Random Forests model demonstrated strong predictive performance, achieving an RMSLE of 0.273 and accuracy score of 0.984 on the test set. Important features such as AGE, ALCOHOL CONSUMING, ALLERGY, PEER PRESSURE, YELLOW FINGERS were identified.

-   **Support Vector Machines (SVM)**: SVMs achieved an RMSLE of 0.346 and accuracy score of 0.952 on the test data. The important features are CHRONIC DISEASE, ALLERGY, FATIGUE, ALCOHOL CONSUMING, ANXIETY.

Overall, the Random Forests model emerged as the top performer, demonstrating robust predictive capabilities for lung cancer prediction. The features ALCOHOL CONSUMING, ALLERGY, FATIGUE are highlighted in most of the models. The least important feature is GENDER(MALE, FEMALE).

## References

-   [Kaggle Notebook - Lung Cancer Analysis](https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4/notebook): A Kaggle notebook providing analysis and insights into lung cancer prediction.
-   [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html): Official documentation for scikit-learn library, used for machine learning models.
