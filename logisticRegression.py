import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def test_data():

    train_df = pd.read_csv('dota2Train.csv', header=None)
    test_df = pd.read_csv('dota2Test.csv', header=None)

    dota2_data = pd.concat([train_df, test_df], ignore_index=True)

    X = dota2_data.drop([0], axis=1)
    y = dota2_data[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    model = LogisticRegression(solver="liblinear", penalty="l1")

    y_score = model.fit(X_train, y_train).predict_proba(X_test)

    model.fit(X_train,y_train)

    predictions = model.predict(X_test) 

    print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

    cf_matrix = confusion_matrix(y_test, predictions)

    print(classification_report(y_test, predictions))

    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Greens')

    plt.show()

test_data()

