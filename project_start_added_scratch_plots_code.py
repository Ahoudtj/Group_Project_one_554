import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#test data
price_test = pd.read_csv('pricing_test.csv', header = None)
price_test = price_test.rename(columns={0: 'sku', 1: 'price', 2: 'quantity', 3: 'order', 4: 'duration', 5: 'category'})
price_test.head()

# training data
price_train = pd.read_csv('pricing.csv', sep = ',')
price_train.head()



# check for outliers to determine 
# plot the skus 
plt.hist(price_train['sku'])
plt.show()
price_train['sku'].describe()
# plot the price
plt.hist(price_train['price'])
plt.show()
price_train.hist(column = 'price')
# plot the order
plt.hist(price_train['order'])
plt.show()
price_train.hist(column = 'order')
# plot the duration
plt.hist(price_train['duration'])
plt.show()
price_train.hist(column = 'duration')
# plot the category
plt.hist(price_train['category'])
plt.show()
price_train.hist(column = 'category')
# plot the quantity
plt.hist(price_train['quantity'])
plt.show()
price_train.hist(column = 'quantity')



# the training data
X1 = np.array(price_train['sku'])
X2 = np.array(price_train['price'])
X3 = np.array(price_train['order'])
X4 = np.array(price_train['duration'])
X5 = np.array(price_train['category'])
X = np.array(np.column_stack((X1, X2, X3, X4, X5))) 
y = np.array(price_train['quantity'])

# the test data
X1_test = np.array(price_test['sku'])
X2_test = np.array(price_test['price'])
X3_test = np.array(price_test['order'])
X4_test = np.array(price_test['duration'])
X5_test = np.array(price_test['category'])
X_test = np.array(np.column_stack((X1, X2, X3, X4, X5))) 
y = np.array(price_test['quantity'])




# the layers:
inputs = tf.keras.layers.Input(shape=(X.shape[1], ), name = 'input') 
hidden1 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=5, activation='sigmoid', name='hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output')(hidden3)



# the model:
model = tf.keras.Model(inputs = inputs, outputs = output)
# here we compile the model, we need to tell the model that this is the 'loss' we will be using, here we use the MSE loss
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))  # we define the learning rate to be a small #
# here we fit the model
model.fit(x=X, y=y, batch_size = 1, epochs = 10) 
# model predict
yhat = model.predict(x = X)


# testing






# plots
from sklearn.inspection import permutation_importance

# variable importance plots
results = permutation_importance(model, X, y, scoring='accuracy')

# Extract the importance scores and the feature names
importance = results.importances_mean
feature_names = [...]

# Plot the importance scores
import matplotlib.pyplot as plt
plt.bar(feature_names, importance)
plt.xticks(rotation=90)
plt.show()




# learning curve
# Compute the moving average of the MSE
window_size = 3
train_ma = np.convolve(train_errors, np.ones(window_size)/window_size, mode='valid')
val_ma = np.convolve(val_errors, np.ones(window_size)/window_size, mode='valid')

# Plot the learning curve
plt.plot(train_sizes[:len(train_ma)], train_ma, 'o-', label='Training error')
plt.plot(train_sizes[:len(val_ma)], val_ma, 'o-', label='Validation error')
plt.xlabel('Number of instances learned')
plt.ylabel('Moving average of MSE')
plt.legend()
plt.show()




# partial dependencies plots
fig, ax = plt.subplots(figsize=(10, 5))
features = [...]
plot_partial_dependence(model, X, features, ax=ax, line_kw={'linewidth': 2})

plt.show()