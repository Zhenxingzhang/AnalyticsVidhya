import pandas as pd
import numpy as np
import math
import csv
from sklearn.preprocessing import LabelEncoder


def classify(x):
    if x['Credit_History'] == 1.0:
        return "Y"
    else:
        return "N"


def data_munging(df):
    df['Self_Employed'].fillna('No',inplace=True)
    df['Gender'].fillna('Male',inplace=True)
    df['Married'].fillna('Yes',inplace=True)
    df['Dependents'].fillna(0,inplace=True)
    df['Loan_Amount_Term'].fillna(360,inplace=True)


    table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

    LoadAmount_MissValue = df[df['LoanAmount'].isnull()].apply(lambda x: table.loc[x['Self_Employed'],x['Education']], axis = 1)

    df['LoanAmount'].fillna(LoadAmount_MissValue, inplace=True)

    # print df.isnull().sum()

    # df['LoanAmount_log'] = df['LoanAmount'].apply(lambda x: math.log(x))

    df['LoanAmount_log'] = np.log(df['LoanAmount'])
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log(df['TotalIncome'])

    var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])

    # print df.dtypes

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])

  #Make predictions on training set:
  predictions = model.predict(data[predictors])

  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])

    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]

    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)

    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])


df = pd.read_csv("train_u6lujuX.csv")
test_df = pd.read_csv("test_Y3wMUE5.csv")

data_munging(df)

df = df[df['Credit_History'].notnull()]


outcome_var = 'Loan_Status'
model = LogisticRegression()

predictor_var = ['Credit_History', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_Area']
# predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']

classification_model(model, df, predictor_var, outcome_var)

test_df['Credit_History'].fillna(0, inplace=True)
data_munging(test_df)
test_data = test_df[predictor_var]

predict_result = model.predict(test_data)

with open('LogisticClassify_on_Credit_History.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(test_df['Loan_ID'], predict_result))

print "Logistic classifier based on Credit History category!"

# predict_test = test_df.apply(classify, axis = 1)
# with open('Classify_on_Credit_History.csv', 'w') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerows(zip(test_df['Loan_ID'],predict_test))
#
# print "Simple classifier based on Credit History category!"