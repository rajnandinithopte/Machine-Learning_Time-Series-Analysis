# Machine Learning: Time Series Analysis
# 🔷 Time Series Classification and Logistic Regression

This project involves **time series classification** using data from the **AReM dataset**, which consists of **sensor readings from human activities**. The main tasks include **feature extraction, binary classification using logistic regression, and multi-class classification using penalized regression techniques**.

---

## **🔷 Libraries Used**
- **NumPy, Pandas** - Data manipulation and feature engineering.
- **Matplotlib, Seaborn** - Data visualization for scatter plots and distribution analysis.
- **SciPy, Bootstrap** - Statistical analysis and confidence interval estimation.
- **Scikit-learn** - Logistic regression, cross-validation, feature selection, and model evaluation.

---

## **🔷 Dataset Description**
- The **AReM dataset** consists of **sensor readings from seven human activities**.
- Each activity contains **multiple instances**, where each instance is a time series of **six sensor readings**:
  - **avg_rss12, var_rss12, avg_rss13, var_rss13, avg_rss23, var_rss23**
- Each time series has **480 time points** per instance.
- **Training and Test Split**:
  - **Training Set**: Excludes first 1-2 instances from each activity.
  - **Test Set**: First 1-2 instances of "bending" activities and first 1-3 instances of others.

---

## **🔷 Steps Taken to Accomplish the Project**

### **🔶 1. Data Preprocessing and Feature Engineering**
- Downloaded the **AReM dataset** containing **sensor readings from seven human activities**.
- Cleaned the dataset to remove inconsistencies and missing values.
- Extracted **time-domain features** for each sensor signal, including:
  - Minimum, Maximum, Mean, Median
  - Standard Deviation, First Quartile, Third Quartile
- Constructed a **new dataset** where each row corresponds to an instance with extracted features.

### **🔶 2. Statistical Analysis**
- Estimated the **standard deviation** of each feature.
- Used **bootstrapping methods** to compute **90% confidence intervals** for feature variability.
- Selected the **three most important features** using domain knowledge and statistical analysis.

### **🔶 3. Binary Classification with Logistic Regression**
- Created a **binary classification task** to distinguish **"bending" activity from others**.
- **Visualized feature distributions** using scatter plots to assess separability.
- Experimented with **different feature transformations** to improve class separation.

### **🔶 4. Experimenting with Time Series Splitting**
- Split each time series into **two equal parts** and repeated the classification process.
- Extended the experiment by splitting time series into **l ∈ {1,2,…,20}** sub-series.
- Used **logistic regression** to classify bending vs. non-bending activities for each split.
- Evaluated different feature selection methods:
  - **P-values from logistic regression coefficients**
  - **Recursive Feature Elimination (RFE)**
  - **Backward feature selection**
  
### **🔶 5. Model Selection and Cross-Validation**
- Applied **5-fold cross-validation** to optimize the parameters **(l, p)**:
  - **l** = number of time series splits
  - **p** = number of selected features
- Used **stratified cross-validation** to handle potential **class imbalances**.

### **🔶 6. Evaluation Metrics**
- Reported:
  - **Confusion Matrix**
  - **ROC Curve and AUC Score**
  - **Optimal logistic regression parameters (βi’s)**
  - **Feature importance and statistical significance**
- Compared **test accuracy** against **cross-validation performance**.

### **🔶 7. Handling Class Imbalance**
- Analyzed class separability to detect possible **instability** in logistic regression.
- If imbalanced classes were found:
  - **Implemented case-control sampling** to balance class representation.
  - Adjusted parameters accordingly and **re-evaluated model performance**.

### **🔶 8. L1-Penalized Logistic Regression**
- Compared **feature selection using p-values** vs. **L1-regularization (LASSO)**.
- Performed **cross-validation for both l (time series splits) and λ (L1 penalty)**.
- Compared **L1-penalized logistic regression** with traditional feature selection methods.

### **🔶 9. Multi-Class Classification**
- Trained an **L1-penalized multinomial regression model** to classify all activities.
- Evaluated performance using **confusion matrices** and **multi-class ROC curves**.
- Compared the logistic regression model against a **Naïve Bayes classifier** using:
  - **Gaussian priors**
  - **Multinomial priors**
- Determined the **best classification method** for this problem.
