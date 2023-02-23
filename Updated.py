import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



#test data
price_test = pd.read_csv('pricing_test.csv', header = None)
price_test = price_test.rename(columns={0: 'sku', 1: 'price', 2: 'quantity', 3: 'order', 4: 'duration', 5: 'category'})
#price_test.head()

# training data
price_train = pd.read_csv('pricing.csv', sep = ',')
price_train.head()

price_train['sku']= np.uint8(price_train['sku'])

price_train= pd.get_dummies(price_train, columns=['sku'])
price_train = pd.get_dummies(price_train, columns=['category'])

#price_train.to_csv('file_name.csv')

#price_train.dtypes
#price_train.describe()
#price_train.isna().sum()
price_train = price_train.dropna()
#price_train.isna().sum()

#len(price_train.columns)

price_train['price']= np.uint8(price_train['price'])
price_train['order']= np.uint8(price_train['order'])
price_train['duration']= np.uint8(price_train['duration'])
price_train['quantity']= np.uint8(price_train['quantity'])


#price_train.iloc[:, 0:4]
#price_train.iloc[:, 4:260]
#price_train.iloc[:, 260:293]



X1 = np.array(price_train.iloc[:, 4:260])
X2 = np.array(price_train['price'])
X3 = np.array(price_train['order'])
X4 = np.array(price_train['duration'])
X5 = np.array(price_train.iloc[:, 260:293])
X = np.array(np.column_stack((X1, X2, X3, X4, X5))) 
y = np.array(price_train['quantity'])


#array_sum = np.sum(X)
#array_has_nan = np.isnan(array_sum)
#print(array_has_nan)



# check for outliers to determine 
# plot the skus 
#plt.hist(price_train['sku'])
#plt.show()
#price_train['sku'].describe()

# plot the price
#plt.hist(price_train['price'])
#plt.show()
#price_train.hist(column = 'price')

# plot the order
#plt.hist(price_train['order'])
#plt.show()
#price_train.hist(column = 'order')

# plot the duration
#plt.hist(price_train['duration'])
#plt.show()
#price_train.hist(column = 'duration')

# plot the category
#plt.hist(price_train['category'])
#plt.show()
#price_train.hist(column = 'category')

# plot the quantity
#plt.hist(price_train['quantity'])
#plt.show()
#price_train.hist(column = 'quantity')








# the layers:
inputs = tf.keras.layers.Input(shape=(X.shape[1], ), name = 'input') 
hidden1 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation='linear', name='output')(hidden3)



# the model:
model = tf.keras.Model(inputs = inputs, outputs = output)
# here we compile the model, we need to tell the model that this is the 'loss' we will be using, here we use the MSE loss
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))  # we define the learning rate to be a small #
#model.compile(loss = ['mse','mse'], loss_weights = [0.5,0.5], optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
# here we fit the model
model.fit(x=X, y=y, batch_size = 1, epochs = 1) 
