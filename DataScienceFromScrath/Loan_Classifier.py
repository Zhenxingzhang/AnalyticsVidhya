#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
import pydot
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv
from sklearn.preprocessing import LabelEncoder

import pdb
#pdb.set_trace()


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

    # var_mod = [ 'Married', 'Dependents', 'Education', 'Self_Employed']
    var_mod_gender= ['GenderM', 'GenderF'] 
    var_mod_property = ['Property_AreaUrban','Property_AreaSemi','Property_AreaRural']
    var_mod = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    #le = LabelEncoder()
    #for i in var_mod:
    #    data[i] = le.fit_transform(data[i])
    #use sparse features
    data['GenderM'] = np.zeros(len(data['Gender'])) 
    data['GenderM'][np.where(data['Gender'] == 'Male')[0]] = 1 

    data['GenderF'] = np.zeros(len(data['Gender']))
    data['GenderF'][np.where(data['Gender'] == 'Female')[0]] = 1 

    data['Property_AreaRural'] = np.zeros(len(data['Gender']))
    data['Property_AreaRural'][np.where(data['Property_Area'] == 'Rural')[0]] = 1 
    data['Property_AreaSemi'] = np.zeros(len(data['Gender']))
    data['Property_AreaSemi'][np.where(data['Property_Area'] == 'Semiurban')[0]] = 1 
    data['Property_AreaUrban'] = np.zeros(len(data['Gender']))
    data['Property_AreaUrban'][np.where(data['Property_Area'] == 'Urban')[0]] = 1 

    le = LabelEncoder()
    for i in var_mod_property+var_mod_gender+var_mod:
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


def main():

    train_data = pd.read_csv("train_u6lujuX.csv")

    loan_table = train_data.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)

    print "munging data..."
    data_munging(train_data, loan_table)
    print "munging data...done"

    predict_data = pd.read_csv("test_Y3wMUE5.csv")
    predict_data['Credit_History'].fillna(1, inplace=True)
    data_munging(predict_data, loan_table)

    train_data = train_data[train_data['Credit_History'].notnull()]



    model = LogisticRegression(C=0.2)
    # model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5)
    #model = RandomForestClassifier(n_estimators=100)
    #model = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)
    # model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
    #model = svm.SVC()

    outcome_var = 'Loan_Status'

    #predictor_var = ['Credit_History', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_Area', 'LoanAmount_log']
    predictor_var = ['Credit_History', 'GenderM','GenderF', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_AreaRural','Property_AreaUrban','Property_AreaSemi', 'LoanAmount_log']
    # predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
    # predictor_var = ['Loan_Amount_Term','LoanAmount_log', 'Credit_History']

    outcome = train_data[outcome_var]

    train_data = train_data[predictor_var]
    test_data = predict_data[predictor_var]

    scaler = StandardScaler()
    scaler.fit(train_data)
    scaler.transform(train_data)
    scaler.transform(test_data)

    print "training..."
    classification_model(model, train_data, outcome)
    print "training done"

    print "testing..."
    predict_result = model.predict(test_data)
    print "testing done"
    write_to_csv("NN_Prediction.csv", predict_data['Loan_ID'], predict_result, "Neural Network Model")

    # predict_test = test_df.apply(classify, axis = 1)
    # write_to_csv("Classify_on_Credit_History.csv", predict_result, "Simple classifier based on Credit History category!")

if __name__ == "__main__":
    main()
