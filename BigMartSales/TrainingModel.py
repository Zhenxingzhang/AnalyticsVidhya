import pandas as pd
train = pd.read_csv('Data/train_modified.csv')
test = pd.read_csv('Data/test_modified.csv')

#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()
print mean_sales

# #Define a dataframe with IDs for submission:
# base1 = test[['Item_Identifier','Outlet_Identifier']]
# base1['Item_Outlet_Sales'] = mean_sales
#
# #Export submission file
# base1.to_csv("Results/baseline.csv", index=False)