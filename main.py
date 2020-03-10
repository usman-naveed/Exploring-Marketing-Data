import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from yellowbrick.regressor import ResidualsPlot

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#reading file into data frame for processcing
df = pd.read_csv('adv_sales.csv')

#removing first column as its just numbering (irrelevant)
df = df.drop(['Unnamed: 0'], axis= 1)

#data exploration and ploting
print(df.head())
print(df.info())
plt.scatter(df['billboard'], df['sales'])
plt.xlabel('Billboard')
plt.ylabel('Sales')
plt.show()
plt.scatter(df['price'], df['sales'])
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()
plt.scatter(df['store'], df['sales'])
plt.xlabel('Store')
plt.ylabel('Sales')
plt.show()
plt.scatter(df['printout'], df['sales'])
plt.xlabel('Printout')
plt.ylabel('Sales')
plt.show()
plt.scatter(df['sat'], df['sales'])
plt.xlabel('Sat')
plt.ylabel('Sales')
plt.show()
plt.scatter(df['comp'], df['sales'])
plt.xlabel('Comp')
plt.ylabel('Sales')
plt.show()

# calculate and plot the correlation matrix
print('Correlation Matrix: \n', df.corr())
labels = ['store', 'store','billboard', 'printout', 'sat', 'comp', 'price', 'sales']
fig, ax = plt.subplots()
im = plt.imshow(df.corr(), cmap='bwr', interpolation='nearest')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.grid(True)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
fig.tight_layout()
plt.colorbar(im)
plt.show()




'''
data is all numeric (int & float) so no need to one hot encode
In terms of data cleaning, we will need to scale/normalize the data 
to get them in the same order of magnitude. This is done below
'''
#scaling data
finalDF = pd.DataFrame(scale(df), columns=list(df))

'''
Creating the linear regression models below
'''
'''
#single variable model
X = finalDF.drop(['store', 'price', 'printout', 'sat', 'comp', 'sales'], axis=1)
y = finalDF['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)
billboard_sales = LinearRegression().fit(X_train, y_train)
bill = billboard_sales.predict(X_test)
plt.scatter(X_test, y_test, edgecolors= 'blue')
plt.plot(X_test, bill, linewidth=3)
plt.title('Single Variable Regression')
plt.xlabel('Billboard')
plt.ylabel('Sales')
plt.show()
visualiser = ResidualsPlot(billboard_sales)
visualiser.score(X_test, y_test)
visualiser.poof()
'''

#first model (M1) using price only
X = finalDF.drop(['store', 'billboard', 'printout', 'sat', 'comp', 'sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m1 = LinearRegression().fit(X_train, y_train)
print('M1 (price): ', m1.score(X_test, y_test))
m1_y = m1.predict(X_test)
plt.scatter(X_test, y_test, edgecolors= 'blue')
plt.plot(X_test, m1_y, linewidth=3)
plt.title('M1')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()
visualiser = ResidualsPlot(m1)
visualiser.score(X_test, y_test)
visualiser.poof()


#second model (M2) using price, store
X = finalDF.drop(['billboard', 'printout', 'sat', 'comp', 'sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m2 = LinearRegression().fit(X_train, y_train)
print('M2 (price, store): ', m2.score(X_test, y_test))
m2_y = m2.predict(X_test)
visualiser = ResidualsPlot(m2)
visualiser.score(X_test, y_test)
visualiser.poof()



#third model (M3) using price, store, billboard
X = finalDF.drop(['printout', 'sat', 'comp', 'sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m3 = LinearRegression().fit(X_train, y_train)
print('M3 (price, store, billboard): ', m3.score(X_test, y_test))
m3_y = m3.predict(X_test)
visualiser = ResidualsPlot(m3)
visualiser.score(X_test, y_test)
visualiser.poof()


#fourth model (M4) using price, store, billboard, printout
X = finalDF.drop(['sat', 'comp', 'sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m4 = LinearRegression().fit(X_train, y_train)
print('M4 (price, store, billboard, printout): ', m4.score(X_test, y_test))
m4_y = m4.predict(X_test)
visualiser = ResidualsPlot(m4)
visualiser.score(X_test, y_test)
visualiser.poof()


#fifth model (M5) using price, store, billboard, printout, satisfaction
X = finalDF.drop(['comp', 'sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m5 = LinearRegression().fit(X_train, y_train)
print('M5 (price, store, billboard, printout, satisfaction): ', m5.score(X_test, y_test))
m5_y = m5.predict(X_test)
visualiser = ResidualsPlot(m5)
visualiser.score(X_test, y_test)
visualiser.poof()


#sixth model (M6) using price, store, billboard, printout, satisfaction, comp
X = finalDF.drop(['sales'], axis = 1)
y = finalDF['sales']

#splitting data into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32)

m6 = LinearRegression().fit(X_train, y_train)
print('M6 (price, store, billboard, printout, satisfaction, comp): ', m6.score(X_test, y_test))
m6_y = m6.predict(X_test)
visualiser = ResidualsPlot(m6)
visualiser.score(X_test, y_test)
visualiser.poof()
