# Concrete Compressing Strength

Concrete is the most important material in civil engineering. Compressive strength or compression strength is the capacity of a material or structure to withstand loads tending to reduce size, as opposed to tensile strength, which withstands loads tending to elongate.

compressive strength is one of the most important engineering properties of concrete. It is a standard industrial practice that the concrete is classified based on grades. This grade is nothing but the Compressive Strength of the concrete cube or cylinder. Cube or Cylinder samples are usually tested under a compression testing machine to obtain the compressive strength of concrete. The test requisites differ country to country based on the design code.

The concrete compressive strength is a highly nonlinear function of age and ingredients .These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.

### Source - http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

# Data Understanding/Exploratory Data Analysis:
Once the data have been collected, we have to understand each and every variable of the data and its characteristics. This can be done by checking number and type of features, descriptive statistics and visualizations, missing values, inconsistent data records etc.

**Shape of the dataset – 1030 Rows & 9 Columns**

**Type of features – 1 Integer &  8 Floats**

**Target Variable - Concrete Compressing Strength (Float)**

### It is a Regression problem since the target variable is continuous.

**Missing Values – No missing values**

**Descriptive Statistics -**
![Descriptive Statistics](https://github.com/SaranyaDScientist/Concrete_Compressing_Strength/blob/master/Concrete_desc_stats.png)

### Data Visualization (Univariate Analysis) - 
From the Univariate analysis, we can understand the central tendency and spread of numerical variables and the proportion of the various levels of categorical variables. Here, numerical variables are analysed through  **Box plots**.

### Outliers -
Outliers are data points that are far from other data points. In other words, they are unusual values in a dataset. In this case, there are unusual values but are not treating them as outliers 

### Data Visualization (Multivariate Analysis) - 

**Pairplot -**
From the **pairplot**, the impact of various X variables on Y variable are visualized, thereby giving clues for feature selection.

![Pairplot](https://github.com/SaranyaDScientist/Concrete_Compressing_Strength/blob/master/Concrete_pairplot.png)

**Heatmap -**
A heatmap is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions. And using **heatmap**, the correlation between the variables are known. From that, we can also find out the highly correlated features.

![Heatmap](https://github.com/SaranyaDScientist/Concrete_Compressing_Strength/blob/master/Concrete_corr.png)

# Modelling:

## CROSS VALIDATION:
Cross validation is a powerful tool that is used for estimating the predictive power of your model, and it performs better than the conventional training and test set. Using cross validation, we can create multiple training and test sets and average the scores to give us a less biased metric.

### K-Fold Cross Validation:
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation. Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

## Linear Regression -

Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x).

When there is a single input variable (x), the method is referred to as simple linear regression. When there are multiple input variables, literature from statistics often refers to the method as multiple linear regression.

Simple linear regression is a regression technique in which the independent variable has a linear relationship with the dependent variable. The straight line in the diagram is the best fit line. The main goal of the simple linear regression is to consider the given data points and plot the best fit line to fit the model in the best way possible.

## Decision Tree Regressor -
Linear regression and logistic regression models fail in situations where the relationship between features and outcome is nonlinear or where features interact with each other. Time to shine for the decision tree! Tree based models split the data multiple times according to certain cut-off values in the features. Through splitting, different subsets of the dataset are created, with each instance belonging to one subset. The final subsets are called terminal or leaf nodes and the intermediate subsets are called internal nodes or split nodes. To predict the outcome in each leaf node, the average outcome of the training data in this node is used. Trees can be used for classification and regression. There are various algorithms that can grow a tree. They differ in the possible structure of the tree (e.g. number of splits per node), the criteria how to find the splits, when to stop splitting and how to estimate the simple models within the leaf nodes. The classification and regression trees (CART) algorithm is probably the most popular algorithm for tree induction. We will focus on CART, but the interpretation is similar for most other tree types.

## K Nearest Neighbour Classifier -

K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. However, it is mainly used for classification predictive problems in industry. The following two properties would define KNN well −

1. Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.

2. Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data.

## Bagging -
Bagging stands for bootstrap aggregation. One way to reduce the variance of an estimate is to average together multiple estimates. Bagging uses bootstrap sampling to obtain the data subsets for training the base learners. For aggregating the outputs of base learners, bagging uses voting for classification and averaging for regression.

## AdaBoost - 
Boosting refers to a family of algorithms that are able to convert weak learners to strong learners. The main principle of boosting is to fit a sequence of weak learners− models that are only slightly better than random guessing, such as small decision trees− to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.
The predictions are then combined through a weighted majority vote (classification) or a weighted sum (regression) to produce the final prediction. The principal difference between boosting and the committee methods, such as bagging, is that base learners are trained in sequence on a weighted version of the data.

AdaBoost is one of the first boosting algorithms to be adapted in solving practices. Adaboost helps you combine multiple “weak classifiers” into a single “strong classifier”. Here are some (fun) facts about Adaboost!

1. The weak learners in AdaBoost are decision trees with a single split, called decision stumps.

2. AdaBoost works by putting more weight on difficult to classify instances and less on those already handled well.

3. AdaBoost algorithms can be used for both classification and regression problem.

## Random Forest Regressor -
A random forest is an ensemble model that consists of many decision trees. Predictions are made by averaging the predictions of each decision tree. This makes random forests a strong modeling technique that’s much more powerful than a single decision tree. Each tree in a random forest is trained on the subset of data provided. The subset is obtained both with respect to rows and columns. This means each random forest tree is trained on a random data point sample, while at each decision node, a random set of features is considered for splitting.

In the realm of machine learning, the random forest regression algorithm can be more suitable for regression problems than other common and popular algorithms. Below are a few cases where you’d likely prefer a random forest algorithm over other regression algorithms:

1. There are non-linear or complex relationships between features and labels.

2. You need a model that’s robust, meaning its dependence on the noise in the training set is limited. The random forest algorithm is more robust than a single decision tree, as it uses a set of uncorrelated decision trees.

3. If your other linear model implementations are suffering from overfitting, you may want to use a random forest.

## Gradient Boosting Regressor -
Gradient boosting regressors are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. 

## Stacked -
Stacking is an ensemble learning technique that combines multiple classification or regression models via a meta-classifier or a meta-regressor. The base level models are trained based on a complete training set, then the meta-model is trained on the outputs of the base level model as features.
The base level often consists of different learning algorithms and therefore stacking ensembles are often heterogeneous. 



