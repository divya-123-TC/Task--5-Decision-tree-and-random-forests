
# Task 5: Decision Trees and Random Forests - All-in-One Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import graphviz

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Decision Tree (Default)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 4. Visualize Decision Tree
plt.figure(figsize=(14, 8))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=class_names)
plt.title("Decision Tree")
plt.show()

# 5. Train Pruned Decision Tree (avoid overfitting)
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_dt_pruned = dt_pruned.predict(X_test)

# 6. Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 7. Accuracy Comparison
print("Decision Tree Accuracy (Default):", accuracy_score(y_test, y_pred_dt))
print("Pruned Decision Tree Accuracy (max_depth=3):", accuracy_score(y_test, y_pred_dt_pruned))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# 8. Feature Importance from Random Forest
importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# 9. Cross-validation
scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
scores_rf = cross_val_score(rf, X, y, cv=5)

print("\nCross-Validation Accuracy:")
print("Pruned Decision Tree: %.2f ± %.2f" % (scores_dt.mean(), scores_dt.std()))
print("Random Forest: %.2f ± %.2f" % (scores_rf.mean(), scores_rf.std()))

# 10. Export Decision Tree as Graphviz PDF (optional)
dot_data = export_graphviz(dt_pruned, out_file=None,
                           feature_names=feature_names,
                           class_names=class_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_pruned")  # saves as PDF














