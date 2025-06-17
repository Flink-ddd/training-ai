# Import libraries  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import load_iris  
from sklearn.tree import DecisionTreeClassifier, export_graphviz  
import graphviz  

# Load dataset  
iris = load_iris()  
X = iris.data  # Features  
y = iris.target  # Labels  
feature_names = iris.feature_names  
class_names = iris.target_names  

# Print dataset details  
print("Dataset shape:", X.shape)  
print("Feature names:", feature_names)  
print("Class names:", class_names) 



# Create a grid of 2D feature plots  
plt.figure(figsize=(12, 10))
plot_index = 1
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        plt.subplot(4, 3, plot_index)  # 4行3列 = 12格
        plt.scatter(X[:, i], X[:, j], c=y, cmap='viridis', edgecolor='k')
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plot_index += 1
plt.tight_layout()
plt.show()



# Split data (for simplicity, use all data for training initially)  
clf = DecisionTreeClassifier(min_samples_leaf=10, random_state=42)  
clf.fit(X, y)  

# Evaluate on training data (note: this is optimistic)  
train_pred = clf.predict(X)  
train_accuracy = np.mean(train_pred == y)  
print(f"Training Accuracy: {train_accuracy:.1%}")  # Output: ~96.0% 



# Export the tree to DOT format and render  
export_graphviz(  
    clf,  
    out_file='iris_tree.dot',  
    feature_names=feature_names,  
    class_names=class_names,  
    filled=True,  
    rounded=True  
)  
with open('iris_tree.dot') as f:  
    dot_graph = f.read()  
graphviz.Source(dot_graph)  



from sklearn.model_selection import cross_val_score  

# 5-fold cross-validation  
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')  
print(f"Cross-validation scores: {scores}")  
print(f"Mean CV Accuracy: {scores.mean():.1%}")  # Output: ~94.7%  



from sklearn.datasets import fetch_openml  

# Load the Seeds dataset
seeds = fetch_openml(name='seeds', version=1, as_frame=False)  
X_seeds, y_seeds = seeds.data, seeds.target.astype(int)  
y_seeds -= 1  # Convert to 0-based labels  
feature_names_seeds = seeds.feature_names  

# 打印部分数据和信息来验证
print("Dataset shape:", X_seeds.shape)
print("Feature names:", feature_names_seeds)
print("First 5 samples:\n", X_seeds[:5])
print("First 5 labels:", y_seeds[:5])




from sklearn.preprocessing import StandardScaler  
from sklearn.pipeline import Pipeline  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report

# Load example dataset (Iris)
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create pipeline: scaling + KNN
clf_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

# Train the model
clf_knn.fit(X_train, y_train)

# Make predictions
y_pred = clf_knn.predict(X_test)

# Evaluate performance
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))





from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import KFold  

# 5-fold cross-validation  
kf = KFold(n_splits=5, shuffle=True, random_state=42)  
accuracies = []  
for train_idx, test_idx in kf.split(X_seeds):  
    X_train, X_test = X_seeds[train_idx], X_seeds[test_idx]  
    y_train, y_test = y_seeds[train_idx], y_seeds[test_idx]  
    clf_knn.fit(X_train, y_train)  
    acc = clf_knn.score(X_test, y_test)  
    accuracies.append(acc)  
print(f"KNN CV Accuracies: {np.round(accuracies, 3)}")  
print(f"Mean Accuracy: {np.mean(accuracies):.1%}")  # Output: ~86.0%  





# Select two features: area (0) and compactness (2)  
X_2d = X_seeds[:, [0, 2]]  
clf_knn_2d = KNeighborsClassifier(n_neighbors=3)  
clf_knn_2d.fit(X_2d, y_seeds)  

# Create meshgrid for plotting  
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1  
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1  
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),  
                     np.linspace(y_min, y_max, 100))  
Z = clf_knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)  

# Plot decision boundary and data points  
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')  
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_seeds, cmap='tab10', edgecolor='k')  
plt.xlabel(feature_names_seeds[0])  
plt.ylabel(feature_names_seeds[2])  
plt.title("KNN Decision Boundary (2D Features)")  
plt.show() 






from sklearn.ensemble import RandomForestClassifier  

# Build a random forest with 100 trees  
rf = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_scores = cross_val_score(rf, X_seeds, y_seeds, cv=5, scoring='accuracy')  
print(f"Random Forest CV Accuracy: {rf_scores.mean():.1%}")  # Output: ~88.0%  







from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names_seeds = iris.feature_names

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize and train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8, 4))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names_seeds[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()






# Final model training (random forest on seeds dataset)  
rf_final = RandomForestClassifier(n_estimators=200, random_state=42)  
rf_final.fit(X_seeds, y_seeds)  
print("Final Model Trained on Full Dataset.") 