#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
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

def write_to_csv(filename, predict_results, message = ""):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Loan_ID", "Loan_Status"])
        writer.writerows(zip(test_df['Loan_ID'], predict_results))
    print message


df = pd.read_csv("train_u6lujuX.csv")
test_df = pd.read_csv("test_Y3wMUE5.csv")

data_munging(df)

df = df[df['Credit_History'].notnull()]


# model = LogisticRegression()
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5)
# model = RandomForestClassifier(n_estimators=100)


# model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)

outcome_var = 'Loan_Status'

# predictor_var = ['Credit_History', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'TotalIncome_log', 'Property_Area', 'LoanAmount_log']
# predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
predictor_var = ['Loan_Amount_Term','LoanAmount_log', 'Credit_History']

#
# Approved = df['Loan_Status'].apply(lambda x : x == 'Y')
# Rejected = df['Loan_Status'].apply(lambda x : x == 'N')
#
# plt.plot(df['Loan_Amount_Term'][Approved], df['LoanAmount_log'][Approved], 'go')
# plt.plot(df['Loan_Amount_Term'][Rejected], df['LoanAmount_log'][Rejected], 'r^')
# # plt.axis([-1, 2, 0, 20])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df['Credit_History'][Approved], df['LoanAmount_log'][Approved], df['Loan_Amount_Term'][Approved],  c='g', marker="o")
# ax.scatter(df['Credit_History'][Rejected], df['LoanAmount_log'][Rejected], df['Loan_Amount_Term'][Rejected],  c='r', marker='^')
# plt.show()

classification_model(model, df, predictor_var, outcome_var)

# dot_data = StringIO()
# export_graphviz(model, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("Loan_Prediction.pdf")

# print model.coef_

# test_df['Credit_History'].fillna(0, inplace=True)
# data_munging(test_df)
# test_data = test_df[predictor_var]
#
# predict_result = model.predict(test_data)
# write_to_csv("RandomForest_Prediction.csv", predict_result, "Random Forest Model")

# predict_test = test_df.apply(classify, axis = 1)
# write_to_csv("Classify_on_Credit_History.csv", predict_result, "Simple classifier based on Credit History category!")