📉 Loan Default Prediction & Data Leakage Analysis

1. Project Objective

The objective of this project is to build a **loan default prediction model** while demonstrating **sound data science judgment**, not just high predictive performance.

This project emphasizes:
Understanding the credit risk domain
Designing financially meaningful features
Identifying and mitigating data leakage
Comparing and selecting a model that is reliable, interpretable, and production-ready

2. Dataset Overview
Source: Kaggle – Non-Performing Assets (NPA) Dataset
Size: ~300,000 loan records

Target Variable - Default_Status
0 → Performing loan
1 → Non-performing loan (default)

Key Input Features
Loan_Amount
Loan_Type (Home, Personal, Education, Vehicle, Business)
Credit_Score (300–850)
Repayment_History (% of on-time payments)
Collateral_Value
Loan_Tenure (months)

The dataset is large enough to expose real-world modeling risks such as leakage and overfitting.

3. Feature Engineering & Financial Rationale

Feature engineering was guided by credit risk logic, not arbitrary transformations.
3.1 Loan Amount per Month (EMI Proxy) : Loan_Amount_per_Month = Loan_Amount / Loan_Tenure

Reasoning:
Borrower stress is driven more by monthly repayment burden than absolute loan size.
This transformation provides a more realistic risk signal than Loan_Amount alone.

3.2 Collateral Coverage Ratio: Collateral_Coverage = Collateral_Value / Loan_Amount
Hypothesis: Lower coverage (<1) should indicate higher default risk.
Finding: Exploratory analysis showed very similar distributions of collateral coverage for defaulters and non-defaulters.
This indicates that collateral is not a dominant driver of default in this dataset, despite financial intuition.

4. Exploratory Data Analysis (EDA) observations
Loan types are almost perfectly balanced, eliminating loan-type sampling bias.
Credit Score and Repayment History show clear separation between defaulters and non-defaulters.
Collateral Coverage does not meaningfully distinguish default behavior.
The dataset is class-imbalanced (~27% defaults), requiring appropriate metrics beyond accuracy.

5. Modeling Strategy

Three models were trained to compare behavior, robustness, and risk:
Logistic Regression
Random Forest
XGBoost

Evaluation Metrics-
ROC–AUC: Measures ranking quality independent of threshold
PR–AUC (Average Precision): Measures quality of high-risk predictions under class imbalance
Confusion Matrix: Operational performance at a fixed threshold
Target Shuffle Test: Explicit check for data leakage

6. Initial Model Results (With All Features)
Observed Performance
Logistic Regression: Test ROC–AUC ≈ 0.74

Random Forest & XGBoost: Test ROC–AUC = 1.00; PR–AUC = 1.00

Interpretation
These results were statistically unrealistic and raised immediate red flags.

7. Data Leakage Detection

To validate whether the high performance was genuine:

7.1 Target Shuffle Test
The target variable was randomly shuffled.
Models were retrained and evaluated.

Result-
Performance collapsed to ~0.5 AUC after shuffling.
This confirmed that the models were exploiting information tied directly to the target.

8. Leakage Source Identification
The feature "Repayment_History" was identified as post-loan behavioral data. This Is Leakage as including it allows models to indirectly conclude the future.

Tree-based models aggressively exploited this signal, resulting in perfect but invalid performance.

9. Clean Modeling (Leakage Removed)
"Repayment_History" was removed, and all models were retrained.
Clean Performance Comparison
| Model               | Test ROC–AUC | PR–AUC | Train ROC–AUC |
| ------------------- | ------------ | ------ | ------------- |
| Logistic Regression | 0.703        | 0.426  | 0.706         |
| Random Forest       | 0.813        | 0.499  | 0.846         |
| XGBoost             | 0.810        | 0.492  | 0.837         |

Validation Checks-
Shuffle test AUC ≈ 0.50 → No leakage
Train vs Test gap is realistic
Confusion matrices show sensible trade-offs

10. Model Interpretation & Feature Importance
Logistic Regression coefficients were stable and directionally reasonable.
Random Forest feature importance highlighted Loan type had relatively low importance, confirming EDA findings.

11. Model Selection Rationale
Although Random Forest and XGBoost achieved higher AUC after leakage removal, Logistic Regression was retained as the primary reference model.
Reasons:
Logistic regression is less prone to exploiting subtle leaked signals.
Coefficient-based explanations align with credit risk governance.
Similar train and test performance indicates controlled variance and stability.

In regulated domains like lending, a slightly weaker but trustworthy model is preferable to a stronger but fragile one.
