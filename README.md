<img width="770" height="782" alt="image" src="https://github.com/user-attachments/assets/9dcdb2d4-71f2-43e5-8d52-26f0a9a5929c" />

# Final Project DATA 3402 - Santander Bank Customer Satisfaction

This project aims to perform a binary classification using Logistic Regression on Santander Customer Satisfaction dataset. We want to predict customer
satisfaction for the bank using data from the [Santander Customer Satisfaction](https://www.kaggle.com/competitions/santander-customer-satisfaction/overview) Kaggle challenge.

## Overview
The task, as defined by the Kaggle challenge, is to predict whether a customer is 
satisfied or unsatisfied with their banking experience at Santander Bank. This repository 
formulates the problem as a binary classification task, using Logistic Regression as the 
model with 143 cleaned and scaled features as input. Our model achieved an AUC-ROC score 
of 0.79 on the test set, which is a solid non-trivial result for a Logistic Regression baseline.

## Summary of Work Done

### Data
- **Type:** CSV file of anonymized numerical features, output: binary satisfaction flag (TARGET column)
- **Size:** 76,020 rows and 371 columns in the training set
- **Split:** 56,943 training samples, 14,236 test samples, no validation set

### Preprocessing / Clean up
- Dropped the `ID` column (row identifier, not a feature)
- Dropped 220 columns with more than 99% zeros
- Replaced `-999999` placeholder values in `var3` with the median
- Dropped 4,841 duplicate rows
- Dropped 6 duplicate columns
- Applied StandardScaler to all features

### Data Visualization
Histograms of selected features were compared between satisfied and unsatisfied customers. 
`var15` showed the clearest difference between classes. 
<img width="826" height="231" alt="image" src="https://github.com/user-attachments/assets/3d16c804-a5d7-4cfc-9665-43c9dcb42fd4" />
Most other features were heavily skewed toward zero making visual separation difficult.

## Problem Formulation
- **Input:** 143 anonymized numerical features
- **Output:** Probability of customer being unsatisfied (TARGET = 1)
- **Model:** Logistic Regression with `class_weight='balanced'` to handle class imbalance
- **Hyperparameters:** `max_iter=1000`, `random_state=42`

## Training
- Trained using scikit-learn on a standard laptop
- Training was fast (under a minute)
- No training curves as Logistic Regression is not an iterative deep learning model

## Performance Comparison
| Metric | Score |
|---|---|
| Accuracy | 0.6926 |
| AUC-ROC | 0.7900 |
| Recall (Unsatisfied) | 0.72 |

## Conclusions
- Logistic Regression achieved an AUC-ROC of 0.79 which is a non-trivial result
- The dataset is heavily imbalanced (24.3:1) which makes accuracy a misleading metric
- `var15` appeared to be the most informative feature visually
- Using `class_weight='balanced'` significantly improved recall for unsatisfied customers

## Future Work
- Try more powerful models like Random Forest or XGBoost
- Perform more sophisticated feature selection
- Try SMOTE or other oversampling techniques to handle class imbalance
- Tune hyperparameters using GridSearchCV

## How to Reproduce Results
1. Clone this repository
2. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data)
3. Place both files in the same directory as the notebook
4. Install required packages (see Software Setup)
5. Run all cells in the notebook in order

## Overview of Files in Repository
- `notebook.ipynb` — Main Jupyter notebook containing all code and analysis
- `submission.csv` — Generated predictions for the Kaggle test set
- `README.md` — This file

## Software Setup
Install required packages: pip install pandas numpy matplotlib scikit-learn

## Data
Download `train.csv` and `test.csv` from the Kaggle competition page:
https://www.kaggle.com/competitions/santander-customer-satisfaction/data

Place both files in the same directory as the notebook before running.

## Citations
- Santander Customer Satisfaction Competition, Kaggle
- scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011
