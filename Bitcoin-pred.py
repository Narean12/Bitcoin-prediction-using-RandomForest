pip install py-AutoClean

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from AutoClean import AutoClean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv("Bitcoin.csv")
pi=AutoClean(df)
pi.output

plt.figure(figsize = (9,5))
sns.heatmap(pi.output.corr(), annot = True,fmt='0.2%')

fig1, axs1 = plt.subplots(4, figsize = (5,5))
plt1 = sns.boxplot(pi.output['Open'], ax = axs1[0])
plt2 = sns.boxplot(pi.output['High'], ax = axs1[1])
plt3 = sns.boxplot(pi.output['Low'], ax = axs1[2])
plt4 = sns.boxplot(pi.output['Volume'], ax = axs1[3])
plt.tight_layout()
plt1.set_title('Open Column')
plt2.set_title('High Column')
plt3.set_title('Low Column')
plt4.set_title('Volumns Column')

sns.pairplot(pi.output,height=3)

pi.output.plot(x='Date', y='Close', color='r')
plt.xticks(rotation=45)
plt.show()

model=RandomForestRegressor(max_depth=1, n_estimators=30,min_samples_split=3, max_leaf_nodes=4, random_state=22)

X=pi.output[['Open','High','Low','Volume']]
X=X[:int(len(df)-1)]
y=pi.output['Close']
y=y[:int(len(df)-1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state = 100)
model.fit(X_train,y_train)

predict=model.predict(X_train)
predict1=model.predict(X_test)
print("y_test shape:", y_test.shape)
print("predict shape:", predict.shape)
res = (y_test - predict)

print("The model score is: ",model.score(X_train,y_train))

warnings.filterwarnings('ignore')

fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Rate', fontsize = 15)
plt.xlabel('Trained Data - Prediction Data', fontsize = 15)
plt.show()

r_squared = r2_score(y_test, predict)
print('R-Squared error: ',r_squared)
print('Mean squared error: %.2f'% mean_squared_error(y_test, predict))
print('Mean Absolute Error: %.2f'% mean_absolute_error(y_test, predict))

new_data=pi.output[['Open','High','Low','Volume']].tail(1)
prediction=model.predict(new_data)
print("The predition of the model: ",prediction)
print("Actual value: ",pi.output[['Close']].tail(1).values[0][0])

plt.figure(figsize=(10, 6))
pi.output['Date'] = pd.to_datetime(pi.output['Date'])
plt.xticks(rotation=45)
plt.plot(pi.output['Date'][:50], y_test[:50], label='Actual', marker='o')
plt.plot(pi.output['Date'][:50], predict1[:50], label='Predicted', marker='x')
plt.xlabel('Timestamp')
plt.ylabel('BTC Value')
plt.title('Actual vs Predicted BTC Values')
plt.legend()
plt.grid(True)
plt.show()

df1=pd.read_csv("BTC-USD.csv")
pq=AutoClean(df1)
new_data1=pq.output[['Open','High','Low','Volume']].tail(1)
tion=model.predict(new_data1)
print("The predition of the model: ",tion)
print("Actual value: ",pq.output[['Close']].tail(1).values[0][0])
