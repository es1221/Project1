import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
import xgboost
from sklearn. linear_model import LogisticRegression as logreg 
from sklearn.model_selection import train_test_split 
from sklearn import svm, preprocessing
from sklearn import datasets, metrics, model_selection
from sklearn.metrics import precision_score, recall_score,roc_curve, RocCurveDisplay, classification_report
from sklearn.svm import SVC
from sklearn. tree import DecisionTreeClassifier 
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score

df = pd.read_csv("adult.try.csv")
 

# decision tree 
# Use one-hot encoding on categorial columns
df = pd.get_dummies(df, columns=[' Workclass', ' Occupation', ' Relationship', ' Race', ' Sex'])
# print(df.head())
# shuffle rows
df = df.sample(frac=1)
# split training and testing data
split_point = int(len(df) * 0.8)  # 80% for training
d_train = df[:split_point]
d_test = df[split_point:]
d_train_att = d_train.drop([' Income'], axis=1)
d_train_gt50 = d_train[' Income']
d_test_att = d_test.drop([' Income'], axis=1)
d_test_gt50 = d_test[' Income']
d_att = df.drop([' Income'], axis=1)
d_gt50 = df[' Income']
# number of income > 50K in whole dataset:
print("Income >50K: %d out of %d (%.2f%%)" % (np.sum(d_gt50), len(d_gt50), 100*float(np.sum(d_gt50)) / len(d_gt50)))
# Income >50K: 7508 out of 30162 (24.89%)

# Range of sample sizes to evaluate
sample_sizes = range(100, len(d_train_att), 1000)

# Lists to store metrics
train_precisions = []
test_precisions = []

for sample_size in sample_sizes:
    # Sample data
    sample_att = d_train_att[:sample_size]
    sample_gt50 = d_train_gt50[:sample_size]

    # Fit a decision tree
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    t = t.fit(sample_att, sample_gt50)

    # Calculate precision on training data
    train_pred = t.predict(sample_att)
    train_precision = precision_score(sample_gt50, train_pred)
    train_precisions.append(train_precision)

    # Calculate precision on testing data
    test_pred = t.predict(d_test_att)
    test_precision = precision_score(d_test_gt50, test_pred)
    test_precisions.append(test_precision)

# Plot metrics
plt.figure(figsize=(10,6))
plt.plot(sample_sizes, train_precisions, label='Train Precision')
plt.plot(sample_sizes, test_precisions, label='Test Precision')
plt.xlabel('Sample Size')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Visualize tree
dot_data = tree.export_graphviz(t, out_file=None, label='all', impurity=False, proportion=True, 
                               feature_names=list(d_train_att), class_names=['less than 50K', 'greater than 50K'],
                               filled=True, rounded=True)
graph = graphviz.Source(dot_data,format="png")
graph.view()

tree.plot_tree(t)

# Generate predictions
d_test_pred = t.predict(d_test_att)

# Print Precision
precision = precision_score(d_test_gt50, d_test_pred)
print('Precision: %f' % precision)

# Print Recall
recall = recall_score(d_test_gt50, d_test_pred)
print('Recall: %f' % recall)

# accuracy
print(t.score(d_test_att, d_test_gt50))

# # Range of sample sizes to evaluate
sample_sizes = range(100, len(d_train_att), 1000)

# Lists to store metrics
accuracies = []
precisions = []
# recalls = []

for sample_size in sample_sizes:
    # Sample data
    sample_att = d_train_att[:sample_size]
    sample_gt50 = d_train_gt50[:sample_size]

    # Fit the decision tree
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    t = t.fit(sample_att, sample_gt50)

    # Generate predictions
    d_test_pred = t.predict(d_test_att)

    # Compute metrics
    accuracy = accuracy_score(d_test_gt50, d_test_pred)
    precision = precision_score(d_test_gt50, d_test_pred)
    recall = recall_score(d_test_gt50, d_test_pred)

    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Plot metrics
plt.figure(figsize=(10,6))
plt.plot(sample_sizes, accuracies, label='Accuracy')
plt.plot(sample_sizes, precisions, label='Precision')
plt.plot(sample_sizes, recalls, label='Recall')
plt.xlabel('Sample Size')
plt.ylabel('Score')
plt.legend()
plt.show()



# Depth of decision tree
for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_gt50, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std()*2))

# Lists to store mean accuracies and depth values
mean_accuracies = []
max_depths = list(range(1, 20))

for max_depth in max_depths:
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_gt50, cv=5)
    mean_accuracies.append(scores.mean())
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std()*2))

# Plot mean accuracy as a function of max depth
plt.figure(figsize=(10,6))
plt.plot(max_depths, mean_accuracies)
plt.xlabel('Max Depth')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy as a Function of Max Depth')
plt.grid(True)
plt.show()

# Generate confusion matrix
confusion_matrix = metrics.confusion_matrix(d_test_gt50, d_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()


# heat map
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title('Adult data set', fontsize=22)
plt.show()

# def barplot():
    dataframe = pd.read_csv("adult.data.Eesha copy.csv")
    dataframe.groupby(['Education'])[' Income'].mean().plot(kind = "bar")
    plt.title('Graph showing' + ' Education' + ' against income')
    plt.xlabel('Education')
    plt.ylabel('Probability of earning more than $50,000')
    plt.show()

barplot()

