# tiny Decision Tree (sklearn) — runs on Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

X,y = load_iris(return_X_y=True)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

clf = DecisionTreeClassifier(min_samples_split=4, random_state=42).fit(Xtr,ytr)
yhat = clf.predict(Xte)
print("acc:", round(accuracy_score(yte,yhat),3))
print(classification_report(yte,yhat,target_names=load_iris().target_names))

# Optional: show tree (uncomment to view)
# plt.figure(figsize=(8,6)); plot_tree(clf, feature_names=load_iris().feature_names, class_names=load_iris().target_names, filled=True); plt.show()
