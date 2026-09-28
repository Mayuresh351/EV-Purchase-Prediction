# EV Purchase Prediction

An end-to-end machine learning project to predict whether a customer is
likely to purchase an electric vehicle (EV), while also identifying the
factors that carry the strongest predictive signal in the dataset.

The project focuses not only on model performance, but also on the
reasoning behind data preparation, model selection, validation and
interpretation.

------------------------------------------------------------------------

## 1. Problem Statement

The objective is to predict the `Will_Buy_EV` target variable using
customer-level demographic, economic, behavioural and EV-infrastructure
related information.

The dataset contains information such as:

-   Age
-   Annual income
-   Daily commute distance
-   Number of cars owned
-   Charging stations near home/work
-   Environmental concern
-   Gender
-   City type
-   Current car type
-   Home charging availability
-   Subsidy availability
-   Range anxiety

The target variable is:

`Will_Buy_EV`

where:

-   `Yes` → customer is willing to buy an EV
-   `No` → customer is not willing to buy an EV

The broader analytical objective is to understand **which factors are
most informative when predicting EV purchase behaviour**.

------------------------------------------------------------------------

# 2. Data Understanding and EDA

The training dataset contains approximately **668K observations** and
multiple numerical and categorical predictors.

The first stage was to understand the structure and quality of the data
before choosing any modelling approach.

### Checks performed

-   Dataset shape
-   Descriptive statistics
-   Null-value checks
-   Data types
-   Numerical vs categorical variables
-   Target distribution
-   Cardinality of categorical variables
-   Train/test dimensions
-   Validity of observed values

No missing values were found in the training data, and based on the
initial inspection there was no identified requirement for imputation or
dropping observations.

### Why is this important?

A machine learning model should not be selected before understanding the
data it will receive. Data quality and feature characteristics directly
influence preprocessing and model selection.

------------------------------------------------------------------------

# 3. Understanding Cardinality

**Cardinality** refers to the number of unique values present in a
feature.

For example:

``` text
Gender
Male
Female
Other
```

has a cardinality of 3.

The categorical variables in this dataset have relatively low
cardinality. This influenced the choice of encoding technique.

### Why does cardinality matter?

Encoding a categorical feature with thousands of unique values is very
different from encoding one with only two or three values.

For low-cardinality categorical variables, **One-Hot Encoding (OHE)** is
a straightforward approach because it converts each category into a
separate binary feature without introducing an artificial numerical
ordering.

------------------------------------------------------------------------

# 4. One-Hot Encoding

Categorical variables cannot be directly interpreted numerically by
XGBoost in the preprocessing setup used here.

One-Hot Encoding converts categories into binary columns.

For example:

``` text
Subsidy_Available
Yes
No
```

becomes:

``` text
Subsidy_Available_Yes
Subsidy_Available_No
```

The project uses:

``` python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)
```

`handle_unknown='ignore'` ensures that an unexpected category in
validation/test data does not cause the transformation to fail.

The numerical features are retained separately and then combined with
the encoded categorical features before model training.

------------------------------------------------------------------------

# 5. Target and Validation Strategy

The target was converted into binary values:

``` text
Yes → 1
No  → 0
```

The data was divided into:

-   **80% training**
-   **20% validation**

A `random_state` of 42 was used for reproducibility.

Importantly, the split was **stratified**:

``` python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why stratification?

Stratification maintains approximately the same target-class proportions
in both the training and validation sets.

This is particularly useful when the target classes are not perfectly
balanced.

------------------------------------------------------------------------

# 6. Why ROC-AUC?

The target is a binary classification problem.

Rather than evaluating only whether the predicted class is correct at
one arbitrary probability threshold, **ROC-AUC** evaluates how well the
model separates positive and negative observations across different
thresholds.

ROC-AUC can be interpreted broadly as a measure of the model's ability
to rank likely EV buyers above non-buyers.

A score of:

-   `0.50` → roughly random ranking
-   `1.00` → perfect separation

The project therefore uses ROC-AUC as the primary validation metric.

------------------------------------------------------------------------

# 7. Model Selection

Several classification approaches were considered during the modelling
process:

-   Logistic Regression
-   Random Forest
-   XGBoost

The final implementation uses **XGBClassifier**.

### Why not simply choose the most complicated model?

Model selection should be empirical rather than based purely on
complexity.

Logistic Regression provides a useful linear baseline, while Random
Forest can capture non-linear relationships through ensembles of
decision trees.

XGBoost was selected for the final model because it is well suited to
structured/tabular data and can capture complex non-linear relationships
and feature interactions.

> Note: Random Forest used for this problem is a
> **RandomForestClassifier**, not a Random Forest regressor, because the
> target is categorical (`Yes`/`No`).

------------------------------------------------------------------------

# 8. Random Forest --- What Is It?

Random Forest is an ensemble machine learning method based on multiple
decision trees.

Instead of relying on one decision tree, it builds many trees using
different samples and feature subsets, then combines their predictions.

For classification, the individual trees collectively determine the
predicted class or probability.

### Why was it considered?

Random Forest is a strong baseline for tabular datasets because it can:

-   capture non-linear relationships
-   model interactions between variables
-   work with mixed feature types after encoding
-   provide feature importance estimates

However, XGBoost was ultimately used for the final model.

------------------------------------------------------------------------

# 9. XGBoost --- What Is It?

**XGBoost (Extreme Gradient Boosting)** is an ensemble method that
builds decision trees sequentially.

Each new tree attempts to improve the errors made by the existing
ensemble.

Unlike Random Forest, where trees are largely built independently and
then combined, gradient boosting builds the model progressively.

XGBoost is particularly effective for many structured/tabular machine
learning problems.

------------------------------------------------------------------------

# 10. Final XGBoost Configuration

The final model uses:

``` python
XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
```

### Important parameters

**`n_estimators=1000`**

The maximum number of boosting trees.

**`learning_rate=0.05`**

Controls how much each new tree contributes to the overall model.

**`max_depth=5`**

Controls the maximum depth of individual trees and therefore their
complexity.

**`subsample=0.8`**

Uses 80% of the training observations for each boosting iteration.

**`colsample_bytree=0.8`**

Uses 80% of the features when constructing each tree.

**`min_child_weight=5`**

Controls the minimum amount of instance weight required in a child node
and helps regulate tree complexity.

------------------------------------------------------------------------

# 11. Machine Learning Pipeline

The project follows an end-to-end workflow:

``` text
Raw Data
   ↓
Data Inspection
   ↓
Data Quality Checks
   ↓
Target Identification
   ↓
Numerical / Categorical Separation
   ↓
Train / Validation Split
   ↓
One-Hot Encoding
   ↓
Combine Numerical + Encoded Features
   ↓
XGBoost Training
   ↓
ROC-AUC Validation
   ↓
Feature Importance Analysis
   ↓
Retrain on Full Training Data
   ↓
Test Predictions
   ↓
Submission
```

This is the project's **machine learning pipeline** in a general
workflow sense. It is not implemented as a single
`sklearn.pipeline.Pipeline` object.

------------------------------------------------------------------------

# 12. Validation Result

The final notebook records the following validation result:

### **ROC-AUC: 0.94164**

This was obtained on the 20% stratified validation set.

The model therefore demonstrated strong ability to distinguish between
customers likely and unlikely to purchase an EV within the validation
data.

------------------------------------------------------------------------

# 13. Feature Importance

After training, XGBoost feature importance was examined to understand
which variables contributed most strongly to the model.

The strongest features in the recorded model were:

  Feature                           Importance
  ------------------------------- ------------
  `Subsidy_Available_No`              0.527570
  `Environmental_Concern_Level`       0.198560
  `Subsidy_Available_Yes`             0.195971
  `Range_Anxiety_Level_Low`           0.030821
  `Range_Anxiety_Level_Medium`        0.010304
  `Annual_Income_USD`                 0.009535

The remaining features had considerably smaller individual importance
values.

An important observation is that the two subsidy-related encoded
features together account for a very large proportion of the model's
reported feature importance.

### What does this mean?

The model suggests that **subsidy availability and environmental concern
contain substantially more predictive signal than many of the
demographic and infrastructure variables in this dataset**.

However, feature importance should not be interpreted as proof of
causation.

For example:

> High feature importance for subsidy availability does not by itself
> prove that subsidies cause customers to purchase EVs.

It means that the feature was highly useful to this trained model when
distinguishing between the target classes.

------------------------------------------------------------------------

# 14. Key Inference

The analysis points towards two particularly strong signals:

### 1. Financial incentives

`Subsidy_Available` dominates the model's feature importance.

This indicates that the presence or absence of an EV subsidy is highly
informative when predicting purchase behaviour in this dataset.

### 2. Environmental concern

`Environmental_Concern_Level` is the second major source of predictive
signal.

This suggests that attitudes towards environmental issues contain
substantial information about EV purchase intention.

### Other factors

Variables such as:

-   Range anxiety
-   Annual income
-   Home charging availability
-   Daily commute
-   Age
-   Charging-station availability
-   City type
-   Current car type

contribute to the model, but their individual reported feature
importance is considerably lower than the dominant subsidy and
environmental-concern features.

------------------------------------------------------------------------

# 15. Business Interpretation

From an analytical perspective, the results suggest that EV adoption
strategies could potentially benefit from considering two dimensions
together:

``` text
Financial Incentive
        +
Environmental Attitude
        ↓
EV Purchase Propensity
```

A company or policymaker could potentially use these signals to
investigate customer segments with different levels of purchase
propensity.

For example, future analysis could investigate whether customers with
high environmental concern but no subsidy availability behave
differently from customers who have access to subsidies.

These interpretations are **dataset-specific associations**, not causal
conclusions.

------------------------------------------------------------------------

# 16. Final Model and Predictions

After validation, the encoder was fitted on the complete training
dataset and the final XGBoost model was trained using all available
training observations.

The final model was then used to generate probability predictions for
the test dataset.

The submission contains:

``` text
id
Will_Buy_EV
```

where `Will_Buy_EV` represents the predicted probability of EV purchase.

------------------------------------------------------------------------

# 17. What Could Be Improved Further?

The current model provides a strong baseline, but several directions
could improve both predictive performance and analytical depth.

### Hyperparameter tuning

Systematically experiment with:

-   `max_depth`
-   `learning_rate`
-   `n_estimators`
-   `min_child_weight`
-   `subsample`
-   `colsample_bytree`
-   regularisation parameters such as `reg_alpha` and `reg_lambda`

Rather than changing many parameters simultaneously, experiments should
be tracked so that improvements can be attributed to specific changes.

### Feature engineering

Potential interaction features could include:

-   Income × Subsidy
-   Environmental Concern × Subsidy
-   Range Anxiety × Charging Availability
-   Daily Commute × Range Anxiety
-   Home Charging + Work Charging
-   Total charging infrastructure availability

These may allow the model to capture relationships that are not
represented directly by the original variables.

### Cross-validation

The current evaluation uses a single train-validation split.

Stratified K-Fold cross-validation could be used to determine whether
the observed ROC-AUC is robust across different splits.

### Model explainability

SHAP could provide more detailed explanations of model behaviour,
including:

-   which features push individual predictions higher
-   which features push them lower
-   how numerical feature values affect predictions
-   interaction effects between important variables

### Deeper EDA

The feature importance results suggest that subsidy and environmental
concern deserve deeper investigation.

For example:

-   EV purchase rate by subsidy availability
-   EV purchase rate by environmental concern
-   Joint effect of subsidy and environmental concern
-   Relationship between range anxiety and charging availability
-   Purchase behaviour across income groups

This would strengthen the transition from **prediction** to **business
insight**.

### Kaggle-specific optimisation

Since this is a competition dataset, further improvements could include
investigating:

-   hidden feature interactions
-   redundant variables
-   target-generation patterns
-   alternative validation strategies
-   carefully engineered features
-   ensemble approaches

However, performance improvements should be validated carefully to avoid
overfitting the validation strategy.

------------------------------------------------------------------------

# 18. Project Takeaway

This project was not treated simply as a model-training exercise.

The workflow was:

**Understand the data → make preprocessing decisions → select an
appropriate evaluation metric → compare modelling approaches → train
XGBoost → evaluate performance → interpret feature importance →
translate findings into business implications.**

The main finding from the final model is that **subsidy availability and
environmental concern were substantially stronger predictive signals
than most other individual variables in the dataset**.

This provides a useful starting point for a deeper question:

> **What combination of financial incentives, customer attitudes and
> infrastructure conditions is most associated with EV adoption?**

That question can be explored further through feature engineering,
explainability, segmentation and more rigorous validation.
