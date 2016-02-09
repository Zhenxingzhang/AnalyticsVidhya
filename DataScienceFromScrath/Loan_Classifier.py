import pandas as pd
import numpy as np
import csv

def classify(x):
    if x['Credit_History'] == 1.0:
        return "Y"
    else:
        return "N"

df = pd.read_csv("train_u6lujuX.csv")
test_df = pd.read_csv("test_Y3wMUE5.csv")

predict_test = test_df.apply(classify, axis = 1)

with open('Classify_on_Credit_History.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows(zip(test_df['Loan_ID'],predict_test))

print "Simple classifier based on Credit History category!"