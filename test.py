#import packages and classes
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor#importing ML classes
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from datetime import date

dd = date.today()
print(dd)

dataset = pd.read_csv("Dataset/food_supply.csv")
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.drop(['Member_number'], axis = 1,inplace=True)

dataset['sold'] = dataset.groupby(["Date"])["itemDescription"].transform('nunique')

dataset['year'] = dataset['Date'].dt.year
dataset['month'] = dataset['Date'].dt.month
dataset['day'] = dataset['Date'].dt.day
dataset.drop(['Date'], axis = 1,inplace=True)

encoder = LabelEncoder()
dataset['itemDescription'] = pd.Series(encoder.fit_transform(dataset['itemDescription'].astype(str)))#encode all str columns to numeric

#class to normalize dataset values
scaler = MinMaxScaler(feature_range = (0, 1))
scaler1 = MinMaxScaler(feature_range = (0, 1))

Y = dataset['sold'].ravel()
Y = Y.reshape(-1, 1)
dataset.drop(['sold'], axis = 1,inplace=True)
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]]

X = scaler.fit_transform(X)
Y = scaler1.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#defining global variables
rsquare = []
mse = []

#function to calculate accuracy and prediction sales graph
def calculateMetrics(algorithm, predict, test_labels):
    mse_error = mean_squared_error(test_labels, predict)
    square_error = r2_score(np.asarray(test_labels), np.asarray(predict))
    rsquare.append(square_error)
    mse.append(mse_error)    
    predict = predict.reshape(-1, 1)
    predict = scaler1.inverse_transform(predict)
    test_label = scaler1.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()    
    print()
    print(algorithm+" MSE : "+str(mse_error))
    print(algorithm+" R2 : "+str(square_error))
    print()
    for i in range(0, 10):
        print("Test Data Demand : "+str(test_label[i])+" Predicted Demand : "+str(predict[i]))
    test_label = test_label[0:300]
    predict = predict[0:300]
    plt.figure(figsize=(5,3))
    plt.plot(test_label, color = 'red', label = 'Test Data Demand')
    plt.plot(predict, color = 'green', label = 'Predicted Data Demand')
    plt.title(algorithm+' Food Demand Prediction Graph')
    plt.xlabel('Test Data Days')
    plt.ylabel('Demand Prediction')
    plt.legend()
    plt.show()    

'''
#now train and plot RandomForest Food demand prediction graph
rf = RandomForestRegressor()
#training RandomForest with X and Y training data
rf.fit(X_train, y_train.ravel())
#performing prediction on test data
predict = rf.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)


#now train and plot DecisionTree Food demand prediction graph
rf = DecisionTreeRegressor()
#training DecisionTree with X and Y training data
rf.fit(X_train, y_train.ravel())
#performing prediction on test data
predict = rf.predict(X_test)
calculateMetrics("Decision Tree", predict, y_test)
'''

data = pd.read_csv("Dataset/food_supply.csv", usecols=['itemDescription'])
encoder = LabelEncoder()
values = pd.Series(encoder.fit_transform(data['itemDescription'].astype(str)))#encode all str columns to numeric
'''
data1 = data.values
scaler = MinMaxScaler(feature_range = (0, 1))
data1 = scaler.fit_transform(data1)
'''
values = values.ravel().reshape(-1, 1)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(values)
data['cluster'] = kmeans.labels_

'''
purchase = [['whole milk']]
purchase = pd.DataFrame(purchase, columns=['Product'])
print(purchase)
purchase['Product'] = pd.Series(encoder.transform(purchase['Product'].astype(str)))#encode all str columns to numeric
purchase = purchase.values
print(purchase.shape)
purchase = scaler.transform(purchase)
print(purchase.ravel()[0])
'''
user_cluster = data[data['itemDescription'] == 'pot plants']['cluster'].values[0]
print(user_cluster)
similar_users = data[data['cluster'] == user_cluster]
recommendations = similar_users['itemDescription'].value_counts().index.tolist()
print(recommendations)




