# Diabetes Diagnosis SVM

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Tools and Methods](#tools-and-methods)
- [Results](#results)
- [Limitations](#limitations)

### Project Overview
This group project aimed to leverage machine learning techniques to identify key predictors for diagnosing diabetes. Through data cleaning, preprocessing, and the implementation of Support Vector Machine (SVM) with regularization, significant predictors such as glucose levels, the diabetes pedigree function, and BMI were identified. Despite challenges such as data imbalance and missing values, the project highlights the potential of machine learning in improving and predicting a diabetes diagnosis.

### Data Sources

The primary dataset used for this analysis is the "Diabetes_Diagnosis_Data.csv" file, which was found on Kaggle. The dataset comprises of 768 records and nine columns, with eight predictors. These predictors include number of pregnancies, glucose level after a tolerance test, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age, alongside one target column denoted as outcome. The dataset specifically focuses on individuals aged 21 years and older, with the outcome column representing whether a patient has diabetes (1) or not (0).

### Tools

- Excel: data inspection
- Python: pandas, numpy, sklean (predictive analysis), matplotlib (visualizaiton)

### Data Cleaning and Preparation

In the initial data preparation phase, we performed the following tasks:

- Data loading and inspection

- Identification of null values. There are no missing values. However, many columns have values of "0" where not clinically possible. These values were treated as missing. To maintain the original distribution of the data and minimize the impact on analysis, the column mean replaced the "0" in columns where appropriate.

- Z-Score Standardization was crucial to prevent skewed model outcomes, particularly in cases where predictor values varied significantly in magnitude.

### Tools and Methods

Regularization techniques, involving adjustments to the box constraint (C) and kernel scale (gamma), were employed to minimize error and control model complexity. Optimal parameters were determined through testing various box constraints ranging from 0.001 to 5.0 and gamma values ranging from 0.05 to 3.0, with the most effective model achieved at a box constraint of 1.0 and a gamma value of 0.8. This regularization process highlighted Glucose, Diabetes Pedigree Function, and BMI as significant predictors for diabetes diagnosis. 

### Results

Subsequently,the best SVM RBF model was trained on the dataset. For class 0, it was found that the precision was 0.93, recall was 0.97, and the f1 score was 0.95. For class 1, it was found that the precision was 0.94, recall was 0.87, and the f1 score was 0.90.

```python
blf = svm.SVC (kernel = 'rbf', C=1.0, gamma = 0.8, probability = True)
blf.fit(X_train,Y_train)

bfunc = blf.predict(X_test)
bscores = classification_report(Y_test,bfunc)
print(bscores)
```

A ROC curve was generated as well. 

```python
fpr_0, tpr_0,_ = roc_curve(Y_test, allresults[:,0], pos_label=0)
roc_auc_0= roc_auc_score(Y_test, 1 - allresults[:,0])

fpr_1,tpr_1,_ = roc_curve(Y_test, allresults[:,1], pos_label=1)
roc_auc_1 = roc_auc_score(Y_test, allresults[:,1])

print ('roc_auc_0: ', roc_auc_0)
print ('roc_auc_1:' , roc_auc_1, '\n')

plt.plot(fpr_0, tpr_0, marker = '.', label= "Class 0", color ='b')
plt.plot(fpr_1, tpr_1, marker ='.', label ='Class 1', color = 'r')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

### Limitations

Some potential limitations of this project include:

1. **Data Quality:** While efforts were made to clean and preprocess the dataset, replacing missing values with column means may not fully capture the true underlying patterns in the data.

2. **Feature Selection:** While Glucose, Diabetes Pedigree Function, and BMI were identified as important predictors, there may be other relevant features not included in the analysis. Exploring additional variables or feature engineering techniques could enhance the model's predictive performance.

3. **Generalizability:** The model's performance metrics were evaluated on a specific dataset, and its generalizability to different populations or healthcare settings may be limited. 

4. **Class Imbalance:** The imbalance between diabetes and non-diabetes cases in the dataset could affect model training and evaluation. 

