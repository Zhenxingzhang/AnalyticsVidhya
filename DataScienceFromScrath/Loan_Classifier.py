#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import ensemble
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from numpy import asarray as ar
from sklearn.metrics import precision_score, recall_score
import xgboost as xgb

import csv
from sklearn.preprocessing import LabelEncoder


def classify(x):
    if x['Credit_History'] == 1.0:
        return "Y"
    else:
        return "N"


def data_munging(data, loan_pivot_table):

    data['Self_Employed'].fillna('No', inplace=True)
    data['Gender'].fillna('Male', inplace=True)
    data['Married'].fillna('Yes', inplace=True)
    data['Dependents'].fillna(0, inplace=True)
    data['Loan_Amount_Term'].fillna(360, inplace=True)

    LoadAmount_MissValue = data[data['LoanAmount'].isnull()].apply(lambda x: loan_pivot_table.loc[x['Self_Employed'], x['Education']], axis = 1)

    data['LoanAmount'].fillna(LoadAmount_MissValue, inplace=True)

    data['LoanAmount_log'] = np.log(data['LoanAmount'])
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['TotalIncome_log'] = np.log(data['TotalIncome'])

    data['GenderM'] = np.zeros(len(data['Gender']))
    data.loc[data['Gender'].apply(lambda x: x == 'Male'), 'GenderM'] = 1

    data['GenderF'] = np.zeros(len(data['Gender']))
    data.loc[data['Gender'].apply(lambda x: x == 'Female'), 'GenderF']= 1

    data['Property_AreaRural'] = np.zeros(len(data['Gender']))
    data.loc[data['Property_Area'].apply(lambda x: x == 'Rural'), 'Property_AreaRural'] = 1
    data['Property_AreaSemi'] = np.zeros(len(data['Gender']))
    data.loc[data['Property_Area'].apply(lambda x: x == 'Semiurban'), 'Property_AreaSemi'] = 1
    data['Property_AreaUrban'] = np.zeros(len(data['Gender']))
    data.loc[data['Property_Area'].apply(lambda x: x == 'Urban'), 'Property_AreaUrban'] = 1

    var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    le = LabelEncoder()
    for i in var_mod:
        data[i] = le.fit_transform(data[i])


# Generic function for making a classification model and accessing performance:
def classification_model(model, data, outcome):

    model.fit(data, outcome)

    # Make predictions on training set:
    predictions = model.predict(data)

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, outcome)
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data.iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = outcome.iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data.iloc[test, :], outcome.iloc[test]))

    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data, outcome)


def write_to_csv(filename, loan_ids, predict_results, message=""):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Loan_ID", "Loan_Status"])
        writer.writerows(zip(loan_ids, predict_results))
    print message


def xgboost_classifier(training_data, labels_numeric, test_data):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['gamma'] = 1
    #param['n_estimators'] = 500
    param['min_child_weight'] = 4
    param['max_depth'] = 5
    param['subsample'] = 0.85
    param['colsample_bytree'] = 0.5
    param['max_delta_step'] = 20
    #param['lambda'] = 10
    num_round = 800

    xg_train = xgb.DMatrix(training_data,label=labels_numeric)
    xg_test = xgb.DMatrix(test_data)

    model = xgb.train(param, xg_train , num_round)

    predict_result = model.predict(xg_train, output_margin = True)

    predict_result = pd.Series(predict_result < 0).replace({True: 'N', False: 'Y'})

    return predict_result

def main():

    train_data = pd.read_csv("train_u6lujuX.csv")

    loan_table = train_data.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

    data_munging(train_data, loan_table)

    # print train_data.describe()

    predict_data = pd.read_csv("test_Y3wMUE5.csv")
    predict_data['Credit_History'].fillna(1, inplace=True)
    data_munging(predict_data, loan_table)

    train_data = train_data[train_data['Credit_History'].notnull()]
    # train_data = train_data[train_data['Credit_History'] == 1.0]
    print train_data['Loan_Status'].value_counts()

    baseline_prediction = train_data.apply(classify, axis=1)
    baseline_score = metrics.accuracy_score(baseline_prediction, train_data['Loan_Status'])
    print "Baseline: {:.3f}%".format(baseline_score*100)
    print confusion_matrix(train_data['Loan_Status'], baseline_prediction)

    model = LogisticRegression(C=0.2,class_weight='balanced')
    # model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5)
    # model = RandomForestClassifier(n_estimators=100)
    # model = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
    # model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
    # model = SVC(C= 1.0, kernel= 'rbf', class_weight={'Y': 4, 'N':1})
    # model = SGDClassifier(class_weight='balanced')
    # model = GaussianNB()

    # params = {'n_estimators': 1200, 'max_depth': 5, 'subsample': 0.5,
    #       'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
    # model = ensemble.GradientBoostingClassifier(**params)

    outcome_var = 'Loan_Status'

    # predictor_var = ['Credit_History', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_Area', 'LoanAmount_log']
    # predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
    # predictor_var = ['Loan_Amount_Term','LoanAmount_log', 'Credit_History']
    predictor_var = ['Credit_History', 'GenderM', 'GenderF', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_AreaRural', 'Property_AreaUrban','Property_AreaSemi','LoanAmount_log']
    # predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']

    outcome = train_data[outcome_var]

    training_data = train_data[predictor_var]
    test_data = predict_data[predictor_var]

    scaler = StandardScaler()
    scaler.fit(training_data)
    scaler.transform(training_data)
    scaler.transform(test_data)

    # classification_model(model, training_data, outcome)
    # predict_result = model.predict(training_data)

    labels_numeric = pd.Series(train_data['Loan_Status'].replace({'Y': 1, 'N': 0}),dtype = "float")
    predict_result = xgboost_classifier(training_data, labels_numeric, training_data)

    print confusion_matrix(train_data['Loan_Status'], predict_result)
    print "Binary Precision Score: {0}".format(precision_score(train_data['Loan_Status'], predict_result, pos_label='N', average=None))
    print "Binary Recall Score: {0}".format(recall_score(train_data['Loan_Status'], predict_result, pos_label='N', average=None))


    # print train_data.apply(lambda x: x['Loan_Status'] == 'N', axis = 1).tolist()
    index = [all(tup) for tup in zip((predict_result == 'Y').tolist(), train_data.apply(lambda x: x['Loan_Status'] == 'N', axis = 1).tolist())]
    # print (predict_result == 'Y').tolist()
    # print train_data.apply(lambda x: x['Loan_Status'] == 'N', axis = 1).tolist()
    # print train_data[index].describe()
    # predict_result = model.predict(test_data)
    # write_to_csv("NN_Prediction.csv", predict_data['Loan_ID'], predict_result, "Neural Network Model")

    # predict_test = test_df.apply(classify, axis = 1)
    # write_to_csv("Classify_on_Credit_History.csv", predict_result, "Simple classifier based on Credit History category!")

if __name__ == "__main__":
    main()