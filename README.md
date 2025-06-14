# Task--5-Decision-tree-and-random-forests
 Decision tree and random forests



 Objective:

To understand and implement tree-based classification models such as Decision Trees and Random Forests, evaluate their performance, analyze overfitting, and interpret feature importances.



 Dataset Used:

Heart Disease Dataset



 Tools & Libraries:
Python

Scikit-learn

Pandas

Matplotlib / Seaborn

Graphviz (for visualizing decision trees)



Tasks Covered:

1. Train a Decision Tree Classifier

Loaded the dataset and performed preprocessing.

Trained a decision tree classifier using sklearn.tree.DecisionTreeClassifier.

Visualized the tree using plot_tree() from sklearn.tree.


 2. Analyze Overfitting & Control Tree Depth

Observed overfitting by training without any depth control.

Re-trained the model using max_depth to prevent overfitting.

Compared accuracy on training and test sets.


 3. Train a Random Forest & Compare Accuracy

Trained a random forest using sklearn.ensemble.RandomForestClassifier.

Compared performance with the decision tree classifier.

Used n_estimators, max_depth, and random_state for tuning.


 4. Interpret Feature Importances

Extracted feature importances from the trained random forest.

Visualized the top features using a bar plot.


 5. Evaluate Using Cross-Validation

Used cross_val_score with 5-fold cross-validation to evaluate both models.

Compared mean accuracy across folds.



Results:

Model	Accuracy (Test Set)	Cross-Validation Accuracy

Decision Tree	~0.84	~0.81
Random Forest	~0.88	~0.86






Visuals:

Decision Tree plotted using plot_tree()

Feature importances plotted using matplotlib



