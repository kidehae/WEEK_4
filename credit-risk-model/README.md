# Credit Scoring Business Understanding

## 1. Basel II Regulatory Accord & Model Interpretability
Under the Advanced Internal Ratings-Based (A-IRB) approach of the Basel II Capital Accord, financial institutions are empowered to calculate their own internal risk parameters—specifically the Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). 

Because these metrics directly dictate the minimum amount of regulatory safety capital a bank must hold to protect against unexpected losses, financial regulators require these models to be highly explainable, transparent, and thoroughly documented. A "black-box" model is unacceptable in this context; risk management teams, compliance officers, and external regulators must be able to auditably trace exactly how an individual borrower’s attributes map to a specific risk score to ensure fair lending practices and prevent systemic financial failure.

## 2. The Necessity and Risks of Proxy Target Variables
In this project, the raw eCommerce transaction dataset from the Xente platform lacks a direct historical "default" label (e.g., whether a user explicitly failed to settle a loan). Therefore, a proxy target variable is strictly necessary to train a predictive credit scoring model. By engineering behavioral metrics such as Recency, Frequency, and Monetary (RFM) value, we can programmatically identify highly disengaged or low-value user patterns to serve as a high-risk (bad customer) proxy.

### Associated Business Risks:
* **Misclassification / False Positives:** Highly active customers who simply changed their shopping habits might be flagged as "high risk," causing the bank to reject potentially profitable credit applicants.
* **Underestimated Default Rates / False Negatives:** High-volume users who mask underlying financial distress could be labeled "low risk," leading the bank to extend unbacked credit limits and suffer unexpected defaults.
* **Concept Drift:** Behavioral data on an eCommerce app changes rapidly due to marketing campaigns, economic shifts, or seasonality, meaning an RFM proxy might lose its predictive power much faster than a standard credit history.

## 3. Trade-offs: Traditional vs. Machine Learning Approaches

| Dimension | Simple Model (e.g., Logistic Regression + WoE) | High-Performance Model (e.g., Gradient Boosting / XGBoost) |
| :--- | :--- | :--- |
| **Interpretability** | **Extremely High.** Outputs clear coefficients and a linear relationship with log-odds that easily map to standard, point-based credit scorecards. | **Low.** Operates as complex, non-linear tree ensembles that require secondary frameworks (like SHAP) to explain decisions. |
| **Performance** | **Moderate.** May struggle to capture complex, multi-variable interactions or non-linear behaviors in the data. | **High.** State-of-the-art accuracy at separating risk categories and handling complex data trends. |
| **Regulatory Acceptance** | **Industry Gold Standard.** Seamlessly accepted by compliance regulators worldwide due to its structural clarity. | **Challenging.** Requires robust Explainable AI (XAI) frameworks and strict justification to satisfy regulatory validation. |