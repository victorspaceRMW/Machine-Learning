from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

folds=10
# hold different classification models in a single dictionary
models = {}
models["RandomForest"]        = RandomForestClassifier()
models["SVM"]         = SVC()

print ("#############################################")

for model_name in models:
    clf=models[model_name]
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    print (clf,classification_report(Y_test,Y_pred))
    print ("#############################################")
