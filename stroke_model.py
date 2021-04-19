import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Data
df = pd.read_csv("healthcare_cleaned.csv")

# # # TO - DO
# 1) Oversampling
# 5) Build Models
# 5.1) Support Vector Machine
# 5.2) Logistic Regression
# 5.3) KNN
# 5.4) Desicion Tree

# 6) GridSearchCV
# 7) Build Model with best settings

# OVERSAMPLING
X = df[["age","hypertension","heart_disease","avg_glucose_level","bmi","gender_encoded","work_type_encoded","Residence_type_encoded","ever_married_encoded","smoking_status_encoded"]]
y = df["stroke"]
smk = SMOTE()
X_sam, y_sam = smk.fit_resample(X,y)

print(X_sam.shape)
print(y_sam.shape)

print(X.shape)
print(y.shape)

# TRAIN TEST SPLIT

X = X_sam[["age","hypertension","heart_disease","avg_glucose_level","bmi","gender_encoded","work_type_encoded","Residence_type_encoded","ever_married_encoded","smoking_status_encoded"]]
y = y_sam
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# LOGISTIC REGRESSION
logModel = LogisticRegression()
logModel.fit(X_train,y_train)
predLogReg = logModel.predict(X_test)

print(classification_report(y_test,predLogReg))
print(confusion_matrix(y_test,predLogReg))

# KNN
knn = KNeighborsClassifier(leaf_size=10, n_neighbors=1)
knn.fit(X_train,y_train)

knnPred = knn.predict(X_test)

print(classification_report(y_test,knnPred))
print(confusion_matrix(y_test,knnPred))

# Decision Tree Classifier
decTree = DecisionTreeClassifier()
decTree.fit(X_train,y_train)

decTreePred = decTree.predict(X_test)

print(classification_report(y_test,decTreePred))
print(confusion_matrix(y_test,decTreePred))

# Grid Search CV
# rfc
rfc = RandomForestClassifier()
parameters = {"n_estimators" : range(10,300,10), "criterion" : ("gini","entropy"), "max_features" : ("auto", "sqrt", "log2")}
gs = GridSearchCV(rfc,parameters,scoring="accuracy")
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_estimator_)

# knn
parameters_knn = {"n_neighbors" : range(1,100,5), "weights" : ("uniform","distance"), "leaf_size" : range(10,100,10)}
gsKnn = GridSearchCV(knn,parameters_knn,scoring="accuracy")
gsKnn.fit(X_train,y_train)
print(gsKnn.best_score_)
print(gsKnn.best_estimator_)


# Random Forrest Classifier
rfc = RandomForestClassifier(criterion="entropy",n_estimators=50)
rfc.fit(X_train,y_train)
rfcPred = rfc.predict(X_test)

print(classification_report(y_test,rfcPred))
print(confusion_matrix(y_test,rfcPred))




