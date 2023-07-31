import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost
import shap
# from sklearn.model_selection import train_test_split
from sklearn. linear_model import LogisticRegression as logreg 
from sklearn.model_selection import train_test_split 
import sklearn 
from sklearn import svm, preprocessing
from sklearn import datasets, metrics, model_selection
from sklearn.metrics import precision_score, recall_score,roc_curve, RocCurveDisplay, classification_report
from sklearn.svm import SVC
from sklearn. tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score


df = pd.read_csv("adult.data.Eesha.2.csv")
 

# #SVM MODEL
# df = pd.read_csv("adult.data.Eesha.csv")
# ml_df = pd.DataFrame()
# ml_df['Age'] = df['Age']
# ml_df['Years of Education'] = df[' Years of Education']
# ml_df['Hours of Work'] = df[' Hours Work per Week']
# ml_df['Gender'] = df[' Sex']
# ml_df['Income'] = df[' Income']

# X = ml_df.drop('Income', axis=1).values
# X = preprocessing.scale(X)
# x = ml_df.drop('Income' , axis=1).values
# x = preprocessing.scale(X)
# y = ml_df['Income'].values
# y = ml_df['Income'].values
# test_size = 10000
# X_train = X[:-test_size]
# y_train = y[:-test_size]
# X_test = X[-test_size:]
# y_test = y[-test_size:]
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# svc = SVC(random_state=42)
# svc.fit(X_train, y_train)
# ax = plt.gca()
# svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
# svc_disp.plot(ax=ax)

# decision tree 
# Use one-hot encoding on categorial columns
df = pd.get_dummies(df, columns=[' Workclass', ' Occupation', ' Relationship', ' Race', ' Sex'])
# print(df.head())
# shuffle rows
df = df.sample(frac=1)
# split training and testing data
d_train = df[:25000]
d_test = df[25000:]
d_train_att = d_train.drop([' Income'], axis=1)
d_train_gt50 = d_train[' Income']
d_test_att = d_test.drop([' Income'], axis=1)
d_test_gt50 = d_test[' Income']
d_att = df.drop([' Income'], axis=1)
d_gt50 = df[' Income']
# number of income > 50K in whole dataset:
print("Income >50K: %d out of %d (%.2f%%)" % (np.sum(d_gt50), len(d_gt50), 100*float(np.sum(d_gt50)) / len(d_gt50)))
# Income >50K: 7508 out of 30162 (24.89%)

# Fit a decision tree
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
t = t.fit(d_train_att, d_train_gt50)

# # Visualize tree
dot_data = tree.export_graphviz(t, out_file=None, label='all', impurity=False, proportion=True, 
                               feature_names=list(d_train_att), class_names=['less than 50K', 'greater than 50K'],
                               filled=True, rounded=True)
graph = graphviz.Source(dot_data,format="png")
graph.view()

tree.plot_tree(t)

# accuracy
print(t.score(d_test_att, d_test_gt50))





# # confusion matrix
# y_pred = clf.predict(X_test)
# confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# cm_display.plot()
# plt.show()

# # confusion matrix
# for X,y in zip(X_test,y_test):
#     # print("Model: " + str(clf.predict([X])[0]) + " Actual: " + str(y))
#     predicted = clf.predict([X])[0]
#     actual = y_test
#     confusion_matrix = metrics.confusion_matrix(actual, predicted, normalize='all')
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# cm_display.plot()
# plt.show()

# predicted = np.random.binomial(1, 0.9, size = 1000)
# print(classification_report(y_test, predicted))


# Accuracy precision recall


# Accuracy = metrics.accuracy_score(actual, predicted)
# Precision = metrics.precision_score(actual, predicted)
# Sensitivity_recall = metrics.recall_score(actual, predicted)
# Specificity = metrics.recall_score(actual, predicted, pos_label=0)
# F1_score = metrics.f1_score(actual, predicted)

# #metrics:
# print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})



# #OLS REGRESSION RESULTS
# import statsmodels.api as sm
# Y = ml_df['Income']
# X = ml_df[['Years of Education', 'Age', 'Hours of Work', 'Gender']]
# X = sm.add_constant(X)
# model = sm.OLS(Y, X)
# results = model.fit()
# print(results.summary())


# logistic regression curve 
# df = pd.read_csv("adult.data.Eesha.csv")

# X = df[[' Occupation']]
# y = df[' Income']
# mylr = logreg()
# mylr.fit(X, y)
# plt.scatter(df[[' Occupation']], df[[' Income']])
# line_x = np.arange(0, 102, 2)
# line_y = 1/(1+np.exp(0.042*(line_x) - 2.804))
# plt.plot(line_x, line_y, '-r')
# plt.title('Logistic Regression Plot of Years of Education against Income more than $50,000')
# plt.xlabel('Years of Education')
# plt.ylabel('Probability of earning more than $50,000')
# plt.show()







# Shapley code - add logistic regression to function

# df = pd.read_csv("adult.data.Eesha.csv")
# # print the JS visualization code to the notebook
# shap.initjs()

# X,y = shap.datasets.adult()
# X_display,y_display = shap.datasets.adult(display=True)

# # create a train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
# d_train = xgboost.DMatrix(X_train, label=y_train)
# d_test = xgboost.DMatrix(X_test, label=y_test)

# params = {
#     "eta": 0.01,
#     "objective": "binary:logistic",
#     "subsample": 0.5,
#     "base_score": np.mean(y_train),
#     "eval_metric": "logloss"
# }
# model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=20)

# # this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)


# # train XGBoost model
# X,y = shap.datasets.adult()
# model = xgboost.XGBClassifier().fit(X, y)

# # compute SHAP values
# explainer = shap.Explainer(model, X)
# shap_values = explainer(X)

# shap.plots.waterfall(shap_values[0], max_display=20)
# # shap.summary_plot(shap_values)
# xgboost.plot_importance(model)
# plt.title("xgboost importance model")
# plt.show()

# shap.summary_plot(shap_values, X_display, plot_type="bar")


# df.drop([' Capital Gain'], axis=1, inplace=True) 
# df.drop([' Capital Loss'], axis=1, inplace=True) 
# df.drop([' Final Weight'], axis=1, inplace=True) 
 
# df.to_csv("adult.data.asha.csv", index = False) 



# heat map
# plt.figure(figsize=(12,10), dpi= 80)
# sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
# plt.title('Adult data set', fontsize=22)
# plt.show()

#bar graphs 



# def linegraph():
#     dataframe = pd.read_csv("adult.data.Eesha copy.csv")
#     dataframe.groupby(['Age'])[' Income'].mean().plot()
#     plt.title('Graph showing' + ' Age' + ' against income')
#     plt.xlabel('Age')
#     plt.ylabel('Probability of earning more than $50,000')
#     plt.xlim(10,70)
#     plt.show()



# def barplot():
#     dataframe = pd.read_csv("adult.data.Eesha copy.csv")
#     dataframe.groupby(['Education'])[' Income'].mean().plot(kind = "bar")
#     plt.title('Graph showing' + ' Education' + ' against income')
#     plt.xlabel('Education')
#     plt.ylabel('Probability of earning more than $50,000')
#     plt.show()




