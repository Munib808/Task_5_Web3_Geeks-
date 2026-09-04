# Final Report — UCI Adult Census Income Classification

## Executive Summary

This project develops and productionizes a machine-learning system for predicting whether an individual earns more than $50K per year using the UCI Adult Census Income dataset.

The target is binary:

- `<=50K` → class 0
- `>50K` → class 1

The Day 5 work uses the saved Day 4 production pipeline and performs final validation, model-behavior analysis, feature interpretation, production inference testing, and artifact packaging.

The selected final model is **XGBoost**.

A probability threshold of **0.35** was selected on the development set by maximizing F1. The untouched final test set was then evaluated using this threshold.

Final test performance:

- Accuracy: **0.8623**
- Precision: **0.6856**
- Recall: **0.7844**
- F1: **0.7317**
- ROC-AUC: **0.9311**
- PR-AUC: **0.8357**
- Brier score: **0.0863**

The final deployable artifact is:

```text
day5_final/final_model.joblib
```

---

# 1. Problem Definition

The goal is to classify an individual's income category using demographic, education, employment, relationship, and financial attributes.

The positive class is:

```text
income > 50K
```

The negative class is:

```text
income <= 50K
```

This is a binary classification problem.

Because the `>50K` class represents only approximately 23.93% of the dataset, accuracy alone is not sufficient to evaluate the model. Precision, recall, F1, ROC-AUC, PR-AUC, and Brier score are therefore reported.

---

# 2. Dataset and Data Preparation

The dataset contains **48,842 rows**.

The prediction variables include:

- age
- workclass
- fnlwgt
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country

The target variable is `income`.

Missing values represented by `?` are converted to `NaN`.

The target is mapped as:

```python
{
    "<=50K": 0,
    ">50K": 1
}
```

---

# 3. Data Splitting

A stratified train/development/test split was used with:

```text
random_state = 42
```

The resulting partitions were:

| Split | Rows | Percentage |
|---|---:|---:|
| Train | 34,188 | 70% |
| Development | 4,885 | 10% |
| Test | 9,769 | 20% |

The positive-class rate was approximately **23.93%** in all three partitions.

This confirms that stratification preserved the target distribution.

---

# 4. Feature Engineering

The Day 4 feature-engineering function is included in the pipeline.

The following features are created.

## Age bucket

Age is converted to:

```text
<25
25-34
35-44
45-54
55-64
65+
```

## Hours bucket

Weekly working hours are converted to:

```text
part-time(<35)
full-time(35-45)
overtime(>45)
```

## Capital-gain features

Two additional variables are created:

```text
capital_gain_flag
log_capital_gain
```

The logarithmic transformation is:

```python
np.log1p(capital_gain)
```

## Capital-loss features

Two additional variables are created:

```text
capital_loss_flag
log_capital_loss
```

## Higher education

A binary indicator is created:

```python
higher_ed = education_num >= 13
```

## Education-hours interaction

The model also receives:

```python
edu_hours_interaction = education_num * hours_per_week
```

This provides the model with an explicit interaction between education and working hours.

---

# 5. Preprocessing

Preprocessing is part of the sklearn pipeline rather than being performed manually.

### Numerical variables

The numerical pipeline uses:

1. Median imputation
2. StandardScaler

### Categorical variables

The categorical pipeline uses:

1. Most-frequent imputation
2. OneHotEncoder
3. `handle_unknown="ignore"`

### Final transformed feature count

The final preprocessed representation contains:

**119 features**

This design allows raw production inputs to pass directly through the same transformations used during training.

---

# 6. Final Model

The final selected classifier is:

**XGBoost**

The saved pipeline shows the following important fitted parameters:

```text
n_estimators       = 463
learning_rate      ≈ 0.0199494
max_depth          = 8
min_child_weight   = 3
tree_method        = hist
objective          = binary:logistic
eval_metric        = logloss
n_jobs              = 1
```

The Day 5 notebook loads the fitted Day 4 artifact instead of retraining the final model.

The uploaded Day 5 notebook does not contain the full Day 4 model-comparison table or every hyperparameter-search result. Therefore, no missing comparison scores are fabricated in this report.

---

# 7. Pipeline Validation

The saved artifact:

```text
day4_final_pipeline.joblib
```

was loaded successfully.

The pipeline structure contains:

```text
FunctionTransformer
      ↓
ColumnTransformer
      ↓
XGBClassifier
```

A five-row sanity test confirmed that the model accepts raw Adult Census rows and returns a two-column probability matrix.

The probability output shape was:

```text
(5, 2)
```

This confirms that preprocessing is integrated into the model artifact.

---

# 8. Leakage and Duplicate-Overlap Analysis

An exact-row fingerprint check was performed.

Results:

| Comparison | Overlap |
|---|---:|
| Train ∩ Dev | 10 |
| Train ∩ Test | 15 |
| Dev ∩ Test | 2 |

Thus, exact duplicate records exist across the splits.

This is an important data-quality limitation.

However, the preprocessing operations are correctly embedded in the pipeline, so the imputers, scaler, and encoders are fitted as part of the training pipeline rather than manually fitted using test data.

For a stricter production experiment, deduplication should be performed before splitting, with the policy clearly documented.

---

# 9. Threshold Selection

The probability threshold was selected using the development set only.

Candidate thresholds from **0.05 to 0.95** were evaluated.

The objective was maximum F1.

The selected threshold was:

# **0.35**

Development-set performance at this threshold:

| Metric | Development |
|---|---:|
| Accuracy | 0.8585 |
| Precision | 0.6705 |
| Recall | 0.8041 |
| F1 | 0.7312 |

The final test set was not used to choose this threshold.

This preserves the role of the test set as an untouched final evaluation set.

---

# 10. Final Test Results

The selected threshold of 0.35 was applied to the final test probabilities.

## Final metrics

| Metric | Score |
|---|---:|
| Accuracy | **0.8623** |
| Precision | **0.6856** |
| Recall | **0.7844** |
| F1 | **0.7317** |
| ROC-AUC | **0.9311** |
| PR-AUC | **0.8357** |
| Brier Score | **0.0863** |

The model correctly classifies approximately 86.23% of test examples.

For the positive class, the model achieves 78.44% recall, meaning it identifies a substantial majority of the people who actually belong to the `>50K` class.

Precision is 68.56%, meaning that among records predicted as `>50K`, approximately 68.56% are actually in the positive class.

The ROC-AUC of 0.9311 indicates strong ranking ability.

The PR-AUC of 0.8357 is also strong and is useful given the class imbalance.

---

# 11. Confusion Matrix

The final confusion matrix is:

| | Predicted <=50K | Predicted >50K |
|---|---:|---:|
| Actual <=50K | 6,590 | 841 |
| Actual >50K | 504 | 1,834 |

Therefore:

- True Negatives = **6,590**
- False Positives = **841**
- False Negatives = **504**
- True Positives = **1,834**

Total errors:

**1,345**

The false-positive rate is **8.6089%**.

The false-negative rate is **5.1592%**.

The number of false positives is larger than the number of false negatives. This is consistent with the lower classification threshold of 0.35, which was selected to improve positive-class recall.

---

# 12. Error Analysis

The model's errors were analyzed using the final test predictions.

The detailed files are:

```text
day5_final/test_predictions.csv
day5_final/error_analysis.csv
```

## False positives

Major false-positive categories include:

- Private workclass: **548**
- Married-civ-spouse: **786**
- Husband relationship: **690**
- White race: **752**
- Male: **730**
- United-States native country: **779**
- HS-grad education: **235**
- Some-college education: **206**
- Bachelors education: **198**

These counts identify where the largest number of false positives occurs.

They should not automatically be interpreted as category-specific error rates because large categories naturally produce more absolute errors.

## False negatives

Major false-negative categories include:

- Private workclass: **328**
- Self-emp-not-inc: **84**
- HS-grad: **212**
- Some-college: **84**
- Bachelors: **76**
- Masters: **37**
- Married-civ-spouse: **336**
- Husband: **299**
- Craft-repair occupation: **104**
- Prof-specialty occupation: **66**

These patterns show that some people with `>50K` income are difficult to distinguish from lower-income individuals based on the available variables.

---

# 13. Subgroup Analysis

The notebook evaluates several demographic and occupational subgroups.

## Sex

| Group | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Male | 6,480 | 0.8279 | 0.6848 | 0.8047 | 0.7399 |
| Female | 3,289 | 0.9301 | 0.6908 | 0.6757 | 0.6832 |

The model has higher recall for males, while females have substantially higher accuracy.

## Race

| Group | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| White | 8,348 | 0.8573 | 0.6908 | 0.7928 | 0.7383 |
| Black | 944 | 0.9068 | 0.6341 | 0.6446 | 0.6393 |
| Asian-Pac-Islander | 304 | 0.8553 | 0.6848 | 0.8077 | 0.7412 |
| Amer-Indian-Eskimo | 104 | 0.8558 | 0.3333 | 0.6667 | 0.4444 |
| Other | 69 | 0.8986 | 0.7000 | 0.6364 | 0.6667 |

Small subgroups should be interpreted cautiously because their sample sizes are much smaller.

## Education

Strong F1 performance is observed for:

- Prof-school: **0.8936**
- Doctorate: **0.8404**
- Masters: **0.8342**
- Bachelors: **0.8139**

Some smaller/lower-education categories show weak F1 values:

- 5th-6th: **0.0000**
- 9th: **0.0000**
- 1st-4th: **0.0000**
- 10th: **0.4000**
- 7th-8th: **0.4545**

These values should be interpreted together with subgroup sample sizes.

---

# 14. Model Interpretation

The final XGBoost model exposes tree-based feature importances.

The most important transformed feature is:

**`marital-status_Married-civ-spouse` — 0.207742**

Other important features include:

| Feature | Importance |
|---|---:|
| marital-status_Married-civ-spouse | 0.207742 |
| higher_ed | 0.082206 |
| relationship_Husband | 0.071559 |
| marital-status_Never-married | 0.065768 |
| capital-gain | 0.043208 |
| log_capital_gain | 0.035428 |
| relationship_Wife | 0.028668 |
| relationship_Unmarried | 0.023209 |
| age_bucket_<25 | 0.021030 |
| capital_gain_flag | 0.019684 |
| education-num | 0.018055 |
| marital-status_Divorced | 0.014620 |
| relationship_Not-in-family | 0.014214 |
| edu_hours_interaction | 0.013712 |
| occupation_Other-service | 0.013428 |
| relationship_Own-child | 0.013213 |
| log_capital_loss | 0.012184 |
| capital-loss | 0.011905 |
| occupation_Farming-fishing | 0.011588 |
| occupation_Exec-managerial | 0.009387 |

The feature-importance results suggest that marital status, relationship structure, education, capital gains, age, and occupation provide substantial predictive signal.

These importances describe model behavior and should not be interpreted as causal effects.

---

# 15. Learning-Curve Diagnostics

The final learning-curve result is:

| Quantity | F1 |
|---|---:|
| Training F1 | **0.7517** |
| Validation F1 | **0.7116** |
| Train-validation gap | **0.0401** |

The gap is noticeable but not extreme.

This indicates some overfitting/generalization gap, while the validation performance remains reasonably close to the training performance.

The learning curve is saved as:

```text
day5_final/learning_curve.png
```

---

# 16. Calibration

The notebook reports the final Brier score as:

**0.0863**

The Brier score evaluates the quality of probabilistic predictions.

A lower value is better.

The probability outputs are also used directly for:

- ROC-AUC
- PR-AUC
- threshold selection
- production inference

Calibration should continue to be monitored if the model is deployed because probability quality can change under data drift.

---

# 17. Production Inference

The final pipeline is designed to accept raw Adult Census records.

The production flow is:

```text
Raw record
    ↓
Feature engineering
    ↓
Missing-value handling
    ↓
Scaling / one-hot encoding
    ↓
XGBoost probability
    ↓
Probability threshold = 0.35
    ↓
Final income class
```

The user does not need to manually calculate engineered features.

The production inference function validates the required raw columns before making predictions.

---

# 18. Unseen Inference Test

Ten examples from the test data were passed through the production inference function.

Representative outputs included:

| Probability >50K | Prediction |
|---:|---:|
| 0.0044 | <=50K |
| 0.0726 | <=50K |
| 0.0025 | <=50K |
| 0.0337 | <=50K |
| 0.5550 | >50K |
| 0.0092 | <=50K |
| 0.8574 | >50K |
| 0.0213 | <=50K |
| 0.2509 | <=50K |
| 0.2129 | <=50K |

Because the production threshold is 0.35:

- 0.5550 becomes `>50K`
- 0.8574 becomes `>50K`
- 0.2509 remains `<=50K`

This confirms that the documented threshold is being applied correctly.

---

# 19. Final Artifact

The final model is saved as:

```text
day5_final/final_model.joblib
```

The notebook performs a save verification after writing the artifact.

The artifact contains the trained preprocessing/model pipeline so that inference can be performed using raw records.

---

# 20. Reproducibility

The completed notebook reports the following environment:

```text
Python        3.13.9
NumPy         2.3.5
Pandas        2.3.3
SciPy         1.16.3
Scikit-learn  1.7.2
XGBoost       3.4.1
Joblib        1.5.2
Matplotlib    3.10.6
```

Random state:

```text
42
```

The exact package versions are also written to:

```text
day5_final/requirements.txt
```

---

# 21. Production Readiness Assessment

## Strengths

### 1. End-to-end pipeline

Feature engineering and preprocessing are contained in the model pipeline.

### 2. Untouched final test set

The classification threshold was selected on development data rather than the final test set.

### 3. Multiple evaluation metrics

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Brier score

### 4. Error analysis

False positives and false negatives were explicitly analyzed.

### 5. Subgroup analysis

Performance was examined across multiple demographic and occupational groups.

### 6. Model interpretation

XGBoost feature importance was extracted and visualized.

### 7. Production inference

Raw inputs can be passed directly to the saved pipeline.

### 8. Artifact verification

The final `.joblib` file is explicitly checked after saving.

---

# 22. Production Risks and Limitations

## Duplicate records

Exact duplicates exist across train, development, and test partitions.

This can make the final test estimate somewhat optimistic because identical records can occur in multiple partitions.

A stricter evaluation should deduplicate the data before splitting.

## Historical data

The dataset represents a historical census context. Real-world production data can differ significantly.

## Fairness

The subgroup analysis demonstrates that performance is not identical across all groups.

A production deployment should include formal fairness analysis and monitoring.

## Threshold choice

The threshold of 0.35 was selected to maximize F1.

A real business application may need a different threshold if false positives and false negatives have different costs.

## Model serialization

The pipeline references the feature-engineering function through Python serialization. For long-term production deployment, the function should be moved into a stable importable module and the artifact rebuilt from that module.

## Monitoring

The current project does not implement full production monitoring.

A production system should monitor:

- Input schema
- Missing-value rates
- Feature drift
- Prediction drift
- Probability drift
- Subgroup performance
- Delayed ground-truth performance
- Model latency
- Model failures

---

# 23. Recommended Next Production Steps

The next engineering steps should be:

1. Deduplicate the raw dataset before creating the production train/test split.
2. Move feature engineering into a version-controlled Python module.
3. Rebuild the model artifact using the stable module.
4. Add automated unit tests for preprocessing and inference.
5. Add input-schema validation.
6. Add experiment tracking.
7. Register the final model in a model registry.
8. Containerize the inference service.
9. Expose the model through a REST API.
10. Add monitoring and drift detection.
11. Monitor subgroup performance.
12. Add model/version rollback.
13. Establish a retraining policy.

---

# 24. Final Conclusion

The final XGBoost model provides strong predictive performance for the Adult Census Income classification task.

At the selected threshold of **0.35**, the final untouched test set produced:

```text
Accuracy   = 0.8623
Precision  = 0.6856
Recall     = 0.7844
F1         = 0.7317
ROC-AUC    = 0.9311
PR-AUC     = 0.8357
Brier      = 0.0863
```

The model identifies **1,834 true positive** cases and **504 false negative** cases, while producing **841 false positives**.

The learning-curve gap of **0.0401 F1** indicates a moderate generalization gap.

The strongest model features include marital status, higher education, relationship, capital gain, education level, age bucket, and occupation.

The project also demonstrates production-oriented practices:

- Reproducible data splitting
- Pipeline-based preprocessing
- Development-only threshold selection
- Untouched final test evaluation
- Confusion-matrix analysis
- Error analysis
- Subgroup evaluation
- Feature-importance analysis
- Learning-curve diagnostics
- Production inference testing
- Artifact serialization
- Dependency version capture

The final model artifact is:

```text
day5_final/final_model.joblib
```

Overall, the project completes the core Day 5 transition from an ML experiment to a documented, reproducible, inference-ready machine-learning artifact, while clearly identifying the remaining data-quality, fairness, serialization, and monitoring work required for a stronger real-world production deployment.
