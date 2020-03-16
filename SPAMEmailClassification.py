from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
import pandas as pd

data = pd.read_csv("dataset.csv")
X = data.values[:, 0:56]
y = data.values[:,57]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Random Forest
rfc = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_features="sqrt", oob_score=True, max_depth=None)
parameters = {
    'n_estimators':[100, 200],
    'max_features':['auto', 'sqrt', 'log2']
}

rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
ac = accuracy_score(y_test, pred_rfc)

# MLP
mlpc = MLPClassifier(hidden_layer_sizes=[30,30,30], max_iter=8000, solver='adam', activation='relu')

# Scaling the data so that the MLP classifier is more accurate
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)

# Combine Random Forest classifier with MLP classifier, using the StackingCVClassifier
lr = LogisticRegression()
scvc = StackingCVClassifier(classifiers=[rfc, mlpc], meta_classifier=lr)

# KNN
knnc = KNeighborsClassifier(n_neighbors=3)
clf = knnc.fit(X_train, y_train)
Y_prediction = knnc.predict(X_test)

# Cross validate
print('5-fold cross validation:')
print(" ")

for crossValidation, title in zip([scvc, knnc],
    ['StackingClassifier',
     'KNN']):

    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    accuracyscores = model_selection.cross_val_score(crossValidation, X, y, cv=5, scoring='accuracy')
    precisionscores = cross_val_score(crossValidation, X, y, cv=cv, scoring='precision_weighted')

    print("Accuracy: %0.5f (+/- %0.2f) [%s]" % (accuracyscores.mean(), accuracyscores.std(), title))
    print("Precision: %0.5f (+/- %0.2f) [%s]" % (precisionscores.mean(), precisionscores.std(), title))

