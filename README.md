# UCI Adult Census Income Classification — Final Production ML Project

## 1. Project Overview

This project builds and productionizes a binary machine-learning classifier for the **UCI Adult Census Income** dataset.

The objective is to predict whether an individual has an annual income:

- `<=50K` → negative class (`0`)
- `>50K` → positive class (`1`)

The Day 5 workflow takes the saved Day 4 production pipeline, validates it on an untouched test set, performs model-behavior and error analysis, interprets the final XGBoost model, validates production inference, and saves the final deployable artifact.

The final selected model is **XGBoost**.

---

## 2. Dataset

### Dataset

**UCI Adult Census Income dataset**

The Day 5 notebook loads the dataset from:

`https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv`

### Dataset size

- Total rows: **48,842**
- Total columns after adding the target helper column: **16**
- Original prediction features: **14**
- Target: `income`

### Target definition

| Original value | Encoded class |
|---|---:|
| `<=50K` | 0 |
| `>50K` | 1 |

The positive class is therefore:

**Income > $50K/year**

### Class distribution

The dataset is imbalanced toward the `<=50K` class.

Positive-class rate:

- Train: **23.93%**
- Development: **23.93%**
- Test: **23.93%**

---

## 3. Train / Development / Test Split

A fixed stratified split was used with:

```text
random_state = 42
```

The final split is:

| Split | Rows | Percentage |
|---|---:|---:|
| Train | 34,188 | 70% |
| Development | 4,885 | 10% |
| Test | 9,769 | 20% |
| **Total** | **48,842** | **100%** |

The final test set is kept separate from threshold selection.

The classification threshold was selected exclusively on the development set and then applied once to the final test set.

---

## 4. Feature Engineering

The pipeline contains a `FunctionTransformer` implementing the Day 4 feature-engineering function.

### Engineered features

#### 4.1 Age bucket

`age` is converted into categorical age groups:

- `<25`
- `25-34`
- `35-44`
- `45-54`
- `55-64`
- `65+`

#### 4.2 Hours-per-week bucket

`hours-per-week` is converted into:

- `part-time(<35)`
- `full-time(35-45)`
- `overtime(>45)`

#### 4.3 Capital-gain flag

```python
capital_gain_flag = capital-gain > 0
```

#### 4.4 Log capital gain

```python
log_capital_gain = np.log1p(capital-gain)
```

This reduces the effect of the highly skewed capital-gain distribution.

#### 4.5 Capital-loss flag

```python
capital_loss_flag = capital-loss > 0
```

#### 4.6 Log capital loss

```python
log_capital_loss = np.log1p(capital-loss)
```

#### 4.7 Higher-education indicator

```python
higher_ed = education-num >= 13
```

This creates a binary indicator for higher educational attainment.

#### 4.8 Education-hours interaction

```python
edu_hours_interaction = education-num * hours-per-week
```

This allows the model to capture interactions between education level and weekly working hours.

---

## 5. Preprocessing Pipeline

All preprocessing is embedded inside the saved sklearn pipeline.

This is important for production because the same transformations used during training are automatically applied during inference.

### Numerical preprocessing

Numerical variables use:

1. Median imputation
2. StandardScaler

Numerical features include:

- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `log_capital_gain`
- `log_capital_loss`
- `edu_hours_interaction`

### Engineered flag preprocessing

The engineered binary indicators use median imputation:

- `capital_gain_flag`
- `capital_loss_flag`
- `higher_ed`

### Categorical preprocessing

Categorical variables use:

1. Most-frequent imputation
2. OneHotEncoder
3. `handle_unknown="ignore"`

Original categorical features include:

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

Engineered categorical features:

- `age_bucket`
- `hours_bin`

### Final transformed feature count

After preprocessing and one-hot encoding:

**119 features**

---

## 6. Model Development

The final selected model is:

**XGBoost (`XGBClassifier`)**

The Day 5 artifact contains the fitted preprocessing and XGBoost model.

The final classifier configuration visible in the saved pipeline includes:

- `objective="binary:logistic"`
- `eval_metric="logloss"`
- `tree_method="hist"`
- `n_estimators=463`
- `learning_rate≈0.0199494`
- `max_depth=8`
- `min_child_weight=3`
- `n_jobs=1`

The Day 5 notebook consumes the already-fitted Day 4 artifact rather than retraining the final model.

> Note: the uploaded Day 5 notebook does not contain the complete Day 4 hyperparameter-search results for every shortlisted model, so this README does not invent missing CV scores or hyperparameters.

---

## 7. Pipeline Sanity Check

The saved pipeline was loaded successfully from:

```text
day4_final_pipeline.joblib
```

The pipeline accepted raw, unprocessed Adult Census rows.

A five-row sanity check produced a probability matrix with shape:

```text
(5, 2)
```

Example positive-class probabilities from the sanity check:

```text
0.0044
0.0726
0.0025
0.0337
0.5550
```

The test confirms that raw input can pass directly through:

```text
Raw Data
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
XGBoost
   ↓
Probability
```

---

## 8. Leakage and Data-Overlap Check

The notebook checks for exact duplicate rows across the three splits.

Detected exact row overlaps:

| Split comparison | Exact overlapping rows |
|---|---:|
| Train ∩ Dev | 10 |
| Train ∩ Test | 15 |
| Dev ∩ Test | 2 |

Therefore, the source dataset contains some exact duplicate records across the independently sampled splits.

This is reported as a **data-quality / split-overlap warning** rather than silently ignoring it.

At the same time, the preprocessing itself is correctly contained inside the sklearn pipeline. Imputers, scalers, and encoders are therefore fitted as part of the training pipeline rather than manually fitted on the test set.

For a stricter production evaluation, a future version should consider deduplicating the raw dataset before splitting, with the deduplication policy documented and applied before model fitting.

---

## 9. Classification Threshold Selection

The default 0.50 probability threshold was not used automatically.

Threshold candidates from:

```text
0.05 through 0.95
```

were evaluated on the **development set only**.

The selected threshold maximized development-set F1.

### Selected threshold

**0.35**

Development-set results at the selected threshold:

| Metric | Development |
|---|---:|
| Accuracy | 0.8585 |
| Precision | 0.6705 |
| Recall | 0.8041 |
| F1 | 0.7312 |

The lower threshold improves positive-class recall at the expense of some precision.

Most importantly, the final test set was not used to choose the threshold.

---

## 10. Final Test Performance

The final threshold of **0.35** was applied to the untouched test set.

### Final metrics

| Metric | Test score |
|---|---:|
| Accuracy | **0.8623** |
| Precision | **0.6856** |
| Recall | **0.7844** |
| F1 | **0.7317** |
| ROC-AUC | **0.9311** |
| PR-AUC | **0.8357** |
| Brier Score | **0.0863** |
| Classification Threshold | **0.35** |

### Interpretation

The model achieves:

- **86.23% accuracy**
- **68.56% precision** for the `>50K` class
- **78.44% recall** for the `>50K` class
- **73.17% F1**
- **0.9311 ROC-AUC**
- **0.8357 PR-AUC**
- **0.0863 Brier score**

The ROC-AUC indicates strong ranking/discrimination performance, while PR-AUC is particularly useful because the positive class is the minority class.

---

## 11. Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `<=50K` | 0.9290 | 0.8868 | 0.9074 | 7,431 |
| `>50K` | 0.6856 | 0.7844 | 0.7317 | 2,338 |
| **Accuracy** | | | **0.8623** | 9,769 |
| Macro average | 0.8073 | 0.8356 | 0.8195 | 9,769 |
| Weighted average | 0.8707 | 0.8623 | 0.8654 | 9,769 |

---

## 12. Confusion Matrix

The final confusion matrix is:

| | Predicted `<=50K` | Predicted `>50K` |
|---|---:|---:|
| Actual `<=50K` | **6,590 TN** | **841 FP** |
| Actual `>50K` | **504 FN** | **1,834 TP** |

### Error counts

- True Negatives: **6,590**
- False Positives: **841**
- False Negatives: **504**
- True Positives: **1,834**

Total errors:

**1,345**

### Error rates

- False-positive rate: **8.6089%**
- False-negative rate: **5.1592%**

There are more false positives than false negatives.

However, the threshold was intentionally selected to favor recall for the positive `>50K` class.

---

## 13. Error Analysis

The notebook saves detailed predictions and error analysis to:

```text
day5_final/test_predictions.csv
day5_final/error_analysis.csv
```

### False-positive patterns

The largest false-positive groups include:

- `workclass = Private`: 548
- `marital-status = Married-civ-spouse`: 786
- `relationship = Husband`: 690
- `race = White`: 752
- `sex = Male`: 730
- `native-country = United-States`: 779
- `education = HS-grad`: 235
- `education = Some-college`: 206
- `education = Bachelors`: 198

The large counts for these categories are partly explained by their high representation in the overall dataset. Counts should therefore not be interpreted as category-specific error rates.

### False-negative patterns

Important false-negative groups include:

- `workclass = Private`: 328
- `workclass = Self-emp-not-inc`: 84
- `education = HS-grad`: 212
- `education = Some-college`: 84
- `education = Bachelors`: 76
- `education = Masters`: 37
- `marital-status = Married-civ-spouse`: 336
- `relationship = Husband`: 299
- `occupation = Craft-repair`: 104
- `occupation = Prof-specialty`: 66

These errors indicate that income cannot be perfectly separated using broad demographic, education, employment, and work-related variables.

---

## 14. Subgroup Performance

The notebook evaluates performance by:

- Sex
- Race
- Education
- Relationship
- Workclass
- Occupation

### Sex

| Group | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Male | 6,480 | 0.8279 | 0.6848 | 0.8047 | **0.7399** |
| Female | 3,289 | 0.9301 | 0.6908 | 0.6757 | **0.6832** |

The model has higher recall for males and higher accuracy for females.

### Race

| Group | N | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| White | 8,348 | 0.8573 | 0.6908 | 0.7928 | 0.7383 |
| Black | 944 | 0.9068 | 0.6341 | 0.6446 | 0.6393 |
| Asian-Pac-Islander | 304 | 0.8553 | 0.6848 | 0.8077 | 0.7412 |
| Amer-Indian-Eskimo | 104 | 0.8558 | 0.3333 | 0.6667 | 0.4444 |
| Other | 69 | 0.8986 | 0.7000 | 0.6364 | 0.6667 |

Small groups should be interpreted cautiously because their estimates have higher statistical uncertainty.

### Education

The model performs particularly strongly for:

- `Prof-school`: F1 **0.8936**
- `Doctorate`: F1 **0.8404**
- `Masters`: F1 **0.8342**
- `Bachelors`: F1 **0.8139**

Performance is weaker for several low-frequency education categories, including:

- `5th-6th`: F1 **0.0000**
- `9th`: F1 **0.0000**
- `1st-4th`: F1 **0.0000**
- `10th`: F1 **0.4000**
- `7th-8th`: F1 **0.4545**

These low-support groups require caution before drawing broad conclusions.

---

## 15. Feature Importance

The final model is an XGBoost classifier, so tree-based feature importance is available.

The top features reported by the notebook are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `marital-status_Married-civ-spouse` | 0.207742 |
| 2 | `higher_ed` | 0.082206 |
| 3 | `relationship_Husband` | 0.071559 |
| 4 | `marital-status_Never-married` | 0.065768 |
| 5 | `capital-gain` | 0.043208 |
| 6 | `log_capital_gain` | 0.035428 |
| 7 | `relationship_Wife` | 0.028668 |
| 8 | `relationship_Unmarried` | 0.023209 |
| 9 | `age_bucket_<25` | 0.021030 |
| 10 | `capital_gain_flag` | 0.019684 |
| 11 | `education-num` | 0.018055 |
| 12 | `marital-status_Divorced` | 0.014620 |
| 13 | `relationship_Not-in-family` | 0.014214 |
| 14 | `edu_hours_interaction` | 0.013712 |
| 15 | `occupation_Other-service` | 0.013428 |
| 16 | `relationship_Own-child` | 0.013213 |
| 17 | `log_capital_loss` | 0.012184 |
| 18 | `capital-loss` | 0.011905 |
| 19 | `occupation_Farming-fishing` | 0.011588 |
| 20 | `occupation_Exec-managerial` | 0.009387 |

### Interpretation

`marital-status_Married-civ-spouse` is substantially more influential than any other single transformed feature.

Education-related variables, capital gains, relationship variables, age bucket, and work-related variables also contribute strongly.

The feature-importance output is **model-based importance**, not causal importance. A high importance value means the feature was useful to the fitted XGBoost model; it does not prove that the feature causes higher income.

---

## 16. Learning Curve

The learning-curve analysis reports:

- Final training F1: **0.7517**
- Final validation F1: **0.7116**
- Train-validation gap: **0.0401**

A gap of approximately 0.04 indicates some generalization gap, but the gap is not extreme.

The learning curve is saved as:

```text
day5_final/learning_curve.png
```

---

## 17. Production Inference

The final pipeline accepts raw Adult Census records.

No manual feature engineering is required during inference.

The production flow is:

```text
Raw input DataFrame
        ↓
Feature engineering
        ↓
Imputation
        ↓
Scaling / One-hot encoding
        ↓
XGBoost
        ↓
P(Income > 50K)
        ↓
Threshold = 0.35
        ↓
Final prediction
```

The inference function validates that all required raw columns are present.

### Required raw input columns

```text
age
workclass
fnlwgt
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
```

### Production output

The inference result contains:

- `probability_>50K`
- `prediction`
- `predicted_income`

---

## 18. Production Inference Test

Ten unseen examples were passed through the final pipeline.

Observed probabilities and predictions:

| Probability >50K | Prediction | Predicted income |
|---:|---:|---|
| 0.0044 | 0 | `<=50K` |
| 0.0726 | 0 | `<=50K` |
| 0.0025 | 0 | `<=50K` |
| 0.0337 | 0 | `<=50K` |
| 0.5550 | 1 | `>50K` |
| 0.0092 | 0 | `<=50K` |
| 0.8574 | 1 | `>50K` |
| 0.0213 | 0 | `<=50K` |
| 0.2509 | 0 | `<=50K` |
| 0.2129 | 0 | `<=50K` |

The production inference test confirms that raw records can be passed directly to the model and that the selected 0.35 threshold is applied to generate the final prediction.

---

## 19. Final Production Artifact

The final deployable artifact is:

```text
day5_final/final_model.joblib
```

The notebook verifies that the artifact was successfully written to disk.

The artifact contains the trained sklearn pipeline, including:

- Feature engineering
- Preprocessing
- XGBoost classifier

The selected threshold is **0.35** and is part of the documented inference contract.

---

## 20. Generated Project Files

The Day 5 workflow generates the following important files:

```text
day5_final/
├── final_model.joblib
├── final_metrics.csv
├── test_predictions.csv
├── error_analysis.csv
├── feature_importance.csv
├── model_metadata.json
├── requirements.txt
├── inference_example.py
├── confusion_matrix.png
├── roc_curve.png
├── precision_recall_curve.png
├── threshold_analysis.png
├── learning_curve.png
└── calibration_curve.png
```

---

## 21. Reproducibility

### Random state

```text
42
```

### Environment used in the completed notebook

```text
Python       3.13.9
NumPy        2.3.5
Pandas       2.3.3
SciPy        1.16.3
Scikit-learn 1.7.2
XGBoost      3.4.1
Joblib       1.5.2
Matplotlib   3.10.6
```

The notebook also writes these exact versions to `requirements.txt`.

---

## 22. How to Run Production Inference

Place the following files together:

```text
final_model.joblib
inference_example.py
```

Then run:

```bash
python inference_example.py
```

The script loads the model and performs prediction on a raw Adult Census example.

The user does not need to manually:

- create age buckets
- create hours buckets
- calculate log features
- create capital-gain/loss flags
- create the education-hours interaction
- impute missing values
- scale numerical features
- one-hot encode categorical features

All of these operations are handled by the saved pipeline.

---

## 23. Important Production Caveat: Serialized Feature-Engineering Function

The saved Day 4 pipeline contains a Python `FunctionTransformer` referencing the `engineer_features` function.

For reliable deployment, the feature-engineering function should ideally live in a stable importable Python module rather than only inside a notebook or `__main__`.

Recommended production structure:

```text
project/
├── src/
│   └── feature_engineering.py
├── models/
│   └── final_model.joblib
├── inference_example.py
├── requirements.txt
├── README.md
└── FINAL_REPORT.md
```

This reduces serialization/import problems when moving the model between environments.

---

## 24. Limitations

1. **Duplicate records exist across splits.**  
   The overlap check found 10 Train/Dev, 15 Train/Test, and 2 Dev/Test exact duplicate rows.

2. **The dataset is historical.**  
   The Adult Census dataset does not necessarily represent current labor-market behavior.

3. **Fairness requires deeper investigation.**  
   Performance differs across sex, race, education, relationship, workclass, and occupation groups.

4. **Feature importance is not causality.**  
   Model importance should not be interpreted as causal influence.

5. **Threshold selection optimizes F1.**  
   A business application may require a different threshold depending on the relative cost of false positives and false negatives.

6. **No business-specific cost matrix is used.**  
   The current threshold is optimized for F1 rather than monetary or operational cost.

7. **Probability calibration should be monitored in production.**  
   The Brier score is reported, but production monitoring should continue after deployment.

8. **Data drift is not covered by the current notebook.**  
   A production service should monitor feature distributions, missingness, prediction distributions, and performance when labels become available.

---

## 25. Recommended Future Improvements

### Data quality

- Deduplicate records before splitting.
- Establish a documented duplicate-handling policy.
- Validate incoming production schemas.

### Model improvements

- Compare additional algorithms under the same evaluation protocol.
- Perform more extensive hyperparameter optimization if required.
- Consider probability calibration as a formal production component.
- Evaluate threshold selection using business-specific costs.

### Fairness

Monitor:

- False-positive rate by subgroup
- False-negative rate by subgroup
- Recall by subgroup
- Precision by subgroup
- Calibration by subgroup

### MLOps

Add:

- Experiment tracking
- Model registry
- Data validation
- Model monitoring
- Drift detection
- Prediction logging
- Automated testing
- CI/CD
- Containerized deployment
- API serving
- Model rollback/versioning

---

## 26. Final Result

The final production-ready XGBoost pipeline achieves:

**Accuracy: 86.23%**  
**Precision: 68.56%**  
**Recall: 78.44%**  
**F1: 73.17%**  
**ROC-AUC: 93.11%**  
**PR-AUC: 83.57%**  
**Brier Score: 0.0863**

using a classification threshold of:

**0.35**

The model is packaged as:

```text
day5_final/final_model.joblib
```

The project therefore completes the main Day 5 productionization workflow:

```text
Data
  ↓
Feature Engineering
  ↓
Preprocessing
  ↓
Trained XGBoost Model
  ↓
Development Threshold Selection
  ↓
Untouched Test Evaluation
  ↓
Error Analysis
  ↓
Feature Interpretation
  ↓
Production Inference
  ↓
Saved Model Artifact
  ↓
Documentation
```
