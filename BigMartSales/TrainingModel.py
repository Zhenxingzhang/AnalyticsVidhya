import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

def baseline(train_data):
    # mean_sales = train_data['Item_Outlet_Sales'].mean()
    # print mean_sales
    return train_data.groupby(['Item_Identifier']).apply(lambda sub: sub['Item_Outlet_Sales'].mean())


def modelfit(model, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    model.fit(dtrain[predictors], dtrain[target])

    #Predict training set:
    dtrain_predictions = model.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(model, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    #Print model report:
    print "\nModel Report"
    print "RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))
    print "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))

    #Predict on testing data:
    dtest[target] = model.predict(dtest[predictors])

    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


def baseline_run(train, test):
    baseline_sale = baseline(train)
    print type(baseline_sale)

    base1 = test[['Item_Identifier', 'Outlet_Identifier']]
    # base1['Item_Outlet_Sales'] = np.zeros(base1.shape[0])

    base1.loc[:, 'Item_Outlet_Sales'] = base1.apply(lambda x:  baseline_sale[x['Item_Identifier']], axis=1)

    #Export submission file
    base1.to_csv("Results/baseline.csv", index=False)


def main():
    train = pd.read_csv('Data/train_modified.csv')
    test = pd.read_csv('Data/test_modified.csv')

    target = 'Item_Outlet_Sales'
    IDcol = ['Item_Identifier', 'Outlet_Identifier']

    predictors = [x for x in train.columns if x not in [target]+IDcol]
    # print predictors
    alg1 = LinearRegression(normalize=True)
    modelfit(alg1, train, test, predictors, target, IDcol, 'Results/LinearRegression.csv')
    coef1 = pd.Series(alg1.coef_, predictors).sort_values()
    coef1.plot(kind='bar', title='Model Coefficients')
    plt.show()

if __name__ == "__main__":
    main()