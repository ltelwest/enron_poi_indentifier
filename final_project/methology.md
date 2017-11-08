# Machine Learning Summary

## Dataset / Question
  1. Do I have enough data?
  2. Can I define a question?
  3. Enough and right features to answer that question?

## Features
### 1. Exploration
  * Inspect for correlations
  * Outlier removal
  * Imputation
  * Cleaning
### 2. Creation
  * Think about it like a human
### 3. Representation
  * Text vectorization
  * Discretization
### 4. Transformation
  * PCA
  * ICA
### 5. Selection
  * KBest
  * Percentile
  * Recursive feature elimination
### 6. Scaling
  * Mean subtraction
  * MinMax scaler
  * Standard scaler

## Algorithms
### 1. Pick an algorithm
#### Non labeled data -> Unsupervised:
  * K-Means clustering
  * Spectral clustering
  * PCA
  * Mixture models
  * EM algorithm
  * Outlier detection
#### Labeled data -> Supervised:
##### Non-ordered or discrete output:
  * Decision Tree
  * Naive Bayes
  * SVM
  * Ensembles
  * k nearest neighbors
  * LDA
  * Logistic regression
##### Ordered or continuous output:
  * Linear regression
  * Lasso regression
  * Decision tree regression
  * SV regression

## Evaluation
### Validate
  * Train/Test split
  * k-fold
  * Visualize
### Pick Metric(s)
  * SSE/r^2
  * Precision
  * Recall
  * F1 score
  * ROC curve
  * Custom
  * Bias/Variance
