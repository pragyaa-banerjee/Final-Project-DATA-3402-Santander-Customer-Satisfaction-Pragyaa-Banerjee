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
- **Size before cleaning:** 76,020 rows and 371 columns
- **Size after cleaning:** 71,179 rows and 143 columns
- **Split:** 56,943 training samples, 14,236 test samples, no validation set
- **Class Imbalance:** The dataset is heavily imbalanced. We have 73,012 satisfied customers (0) 
vs 3,008 unsatisfied customers (1), a ratio of 24.3:1. This was addressed during 
training using `class_weight='balanced'` in Logistic Regression, which helped us assign higher 
weight to the minority class (unsatisfied customers)

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
<img width="1004" height="392" alt="image" src="https://github.com/user-attachments/assets/f3746004-51ad-40c3-88d0-eba9560bd61a" />
Most other features were heavily skewed toward zero making visual separation difficult.
A correlation heatmap of the first 20 features revealed strong positive correlations among `imp_op_var39` and `imp_op_var41` 
related features, and strong negative correlations among `ind_var5_0` and `ind_var5`. 

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

<img width="824" height="606" alt="image" src="https://github.com/user-attachments/assets/74084bbe-b18b-4074-8378-f4a935e1d173" />
The ROC curve shows our model (AUC = 0.79) performs significantly better than 
random guessing (AUC = 0.50), indicated by the blue curve staying well above the dashed line.

## Conclusions
- Logistic Regression achieved an AUC-ROC of 0.79 which is a non-trivial result
- The dataset is heavily imbalanced (24.3:1) which makes accuracy a misleading metric — 
in a real banking context, missing an unsatisfied customer is more costly than 
falsely flagging a satisfied one, so recall matters more than accuracy
- `var15` (likely representing customer age) appeared to be the most informative 
feature, suggesting that age may play a role in customer satisfaction at Santander Bank
- Using `class_weight='balanced'` significantly improved recall for unsatisfied 
customers, catching 72% of them — in a real world scenario this means the bank 
could proactively reach out to 72% of at-risk customers before they leave
- Early identification of unhappy customers allows banks to take proactive steps 
such as personalized offers or improved service, potentially reducing customer churn

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
- `Notebook for Final Project DATA 3402 Pragyaa Banerjee.ipynb` — Main Jupyter notebook containing all code and analysis
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
