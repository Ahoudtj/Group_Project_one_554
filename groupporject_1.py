import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



#test data
#price_test = pd.read_csv('pricing_test.csv', header = None)
#price_test = price_test.rename(columns={0: 'sku', 1: 'price', 2: 'quantity', 3: 'order', 4: 'duration', 5: 'category'})
#price_test.head()

# training data
price_train = pd.read_csv('pricing.csv', sep = ',')
price_train.head()

price_train.dtypes
price_train.describe()
price_train.isna().sum()
price_train = price_train.dropna()
price_train.isna().sum()

n = price_train.groupby(["sku","category"],as_index=False)["order"].count()



# # check for outliers to determine 
# # plot the skus 
# plt.hist(price_train['sku'])
# plt.show()
# price_train['sku'].describe()

# # plot the price
# plt.hist(price_train['price'])
# plt.show()
# price_train.hist(column = 'price')

# # plot the order
# plt.hist(price_train['order'])
# plt.show()
# price_train.hist(column = 'order')

# # plot the duration
# plt.hist(price_train['duration'])
# plt.show()
# price_train.hist(column = 'duration')

# # plot the category
# plt.hist(price_train['category'])
# plt.show()
# price_train.hist(column = 'category')

# # plot the quantity
# plt.hist(price_train['quantity'])
# plt.show()
# price_train.hist(column = 'quantity')


#train the model for smaller data set by ramdom sample 1% of the data
price_train = price_train.sample(frac =.01)




# the data
import numpy as np
X1 = np.array(price_train['sku'])
X2 = np.array(price_train['price'])
X3 = np.array(price_train['order'])
X4 = np.array(price_train['duration'])
X5 = np.array(price_train['category'])
X = np.array(np.column_stack((X1, X2, X3, X4, X5))) 
y = np.array(price_train['quantity'])



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

# here we fit the model
model.fit(x=X, y=y, batch_size = 1, epochs = 2) 

yhat = model.predict(x=X)  
#this is the line that we fit our test data to know the prediction. 

model.evaluate(x=X,y=y)

#A learning curve plot


#Partial dependence plot 
ind = 1 #toggle this between 1 to get partial plot for X1 and 2 to get partial plot for X2

#step 2.1
v = np.unique(X[:,ind])

#step 2.2
means = []
for i in v:
    #2.2.1: create novel data set where variable only takes on that value
    cte_X_cp = np.copy(X)
    cte_X_cp[:,ind] = i
    #2.2.2 predict response
    yhat = model.predict(cte_X_cp)
    #2.2.3 mean
    means.append(np.mean(yhat))

    
import matplotlib.pyplot as plt    

plt.plot(t, means)
plt.show()


def mse_fn(y_true,y_pred):
    return (y_true - y_pred)**2

plt.plot(y,mse_fn(y,yhat), label= 'MSE')
plt.title('Loss for y = 0 and different values of yhat')
plt.xlabel('yhat')
plt.ylabel('Loss')
plt.legend()
# plt.show()


#---------- record model time ------------------------
import time
start_time = time.time()
#put the model here
print("--- %s seconds ---" % (time.time() - start_time))

#---------Information ram useage-----

#pip install psutil
import os, psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes 

#or
import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)