import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def model_processing():
    data = pd.read_csv("dataset_admissions.csv")
    dummy_rank = pd.get_dummies(data['rank'], prefix="rank")
    collumns_to_keep = ['admit', 'gre', 'gpa']
    data = data[collumns_to_keep].join(dummy_rank[['rank_1', 'rank_2', 'rank_3', 'rank_4']])
    majority = data[data['admit'] == 0]
    minority = data[data['admit'] == 1]
    minority_upsample = resample(minority, replace=True, n_samples=273, random_state=123)
    new_data = pd.concat([majority, minority_upsample])
    X = new_data.drop("admit", axis=1)
    Y = new_data["admit"]
    X_train, X_test, Y_train, Y_real = train_test_split(X, Y, test_size=0.2, random_state = 11)

    # Logistic
    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(X_train, Y_train)
    y_pred = lr_model.predict(X_test)
    acc_lg = metrics.accuracy_score(Y_real, y_pred)

    # DecisionTree
    from sklearn import tree
    tree_model = tree.DecisionTreeClassifier(max_depth=3)
    tree_model.fit(X_train, Y_train)
    acc_dt = tree_model.score(X_test, Y_real)

    # SVM - SVC
    model_SVC = svm.SVC(kernel="linear")
    model_SVC.fit(X_train, Y_train)
    Y_pred = model_SVC.predict(X_test)
    acc_svc = metrics.accuracy_score(Y_pred, Y_real)

    # RandomForest
    model_random_forest = RandomForestClassifier().fit(X_train, Y_train)
    y_pred_random_forest = model_random_forest.predict(X_test)
    acc_rf = metrics.accuracy_score(Y_real, y_pred_random_forest)

    # Plot
    plt.clf()
    plt.title("Accuracy Comparison Graph")
    plt.ylabel("Accuracy Score")
    plt.xlabel("Machine Learning Algorithms - 1.Logistic Regression / 2.Decision Tree / 3.SVM-SCV / 4.Random Forest")
    x = [acc_lg, acc_dt, acc_svc, acc_rf]
    plt.plot([1, 2, 3, 4], x, color="black")
    plt.scatter(1, acc_lg, marker="o", color="pink", label="Logistic Regression")
    plt.scatter(2, acc_dt, marker="o", color="green", label="Decision Tree")
    plt.scatter(3, acc_svc, marker="o", color="red", label="SVM-SVC")
    plt.scatter(4, acc_rf, marker="o", color="blue", label="Random Forest")
    plt.legend()
    plt.savefig('foo.png')

    models_all = {'LogisticRegression': lr_model,
                  'DecisionTree': tree_model,
                  'SVC': model_SVC,
                  'RandomForest': model_random_forest
                  }
    return models_all

