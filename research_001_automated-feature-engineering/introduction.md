# Adaptive Representation Systems For Structured Data.

Despite advances in automated representation learning, structured/tabular machine learning workflows still rely heavily on manual or semi-manual feature engineering and selection.


## The Problem Statement:

In Chapter 4 (Machine Learning) of "Lineage" AI project, it is clear that machine learning has two bottlenecks:
1. Feature Engineering (selection and importance)
2. Hyperparameter Tuning

## The Question:

Can machine learning systems autonomously discover compact, information-preserving, and non-redundant feature representations from raw structured data?

## The Thinking:

1. Manual feature engineering is a crucial part of machine learning workflow. It determines what gets used and by how much. But feature importance is model-dependent, as different models capture different structures, different interactions, different geometries. This means no universal 'best features'. 
But is this how it is in the real world? Are features just really randomly aligned, or do structured datasets contain latent invariant representations that persist across model families? Is there a way to get a universal feature selection and importance, at least for the selection part? 
For example, in linear models, Ridge regression uses L2 regularization (applies a squared penalty to coefficient that reduces weight to almost zero, but not exact zero). L2 regularization gives feature importance, but still uses all features. 
While Lasso regression uses L1 regularization (applies an absolute-value penalty to coefficients directly), hence turning some features' weights to exactly zero this leads to a direct feature selection. Meaning some features are dropped in Lasso regression, unlike Ridge.
Can we find a way to discover these most informative features in our machine learning workflow by filtering out irrelevant or "noisy" features from our dataset and finding/identifying truly informative features while preserving underlying information structure?
2. Hyperparameter tuning governs the model’s capacity and behavior (number of trees, learning rate, regularization strength, etc.,), which indirectly influences how the model uses input features. Tuning model parameters allows for better model performance (like convergence).   

## The Consideration:

1. Manual feature engineering can be tricky and time consuming, but methods exist to automate this process (like AutoML). The question is: How can we identify and prioritize highly informative, non-redundant features, thereby reducing overfitting (the curse of dimensionality) and increasing analysis speed in production?
2. Hyperparameter tuning can be coupled with nested cross-validation to avoid overfitting the tuning step, while the search itself can be automated (e.g., grid, random, Bayesian). 

## The Plan:

The current idea is to combine: 
1. Statistical Methods
2. Software Engineering (adaptive optimization algorithms, modular feature abstraction frameworks, etc)
3. Model-based importance
4. Dimensionality reduction (optional?)
to extract model-agnostic informative representations that we can feed to all models. 
* Here, we meant features that are robust across a family of models common to structured data workflows, not necessarily a single feature set optimal for every conceivable algorithm.

This extraction plan can be either: 
1. Automatic feature selection from existing set
2. Automatic creation of new features 
3. Both

## The Concern:

1. While selecting features (with the belief of selecting only relevant ones), what if we drop important ones? An additional consideration is maintaining interpretability while learning compact representations.
2. What about the issue of multicollinearity, and derived features that are factors of other features? Additionally, some features may exhibit weak independent importance while contributing strongly through interaction effects with other variables.