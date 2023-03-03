import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# csv.reader

# gen = pd.read_csv('pricing_test copy.csv', sep = ',',chunksize=1)

# gen

# for val in gen:
#     minval_1 = val['price']
#     minval_2 = val["duration"]
#     minval_3 = val["order"]
#     print( minval_1)


# minvalue_1=999999999999999.999999999999 #price
# minvalue_2=999999999999999.999999999999 #quantity
# minvalue_3=999999999999999.999999999999 #order
# minvalue_4=999999999999999.999999999999 #duration 
# maxvalue_1=0 #price
# maxvalue_2=0 #quantity
# maxvalue_3=0 #order
# maxvalue_4=0 #duration 
# category = []
# for val in gen:
#     minvalue_1 = min(minvalue_1,float(val["price"]))
#     minvalue_2 = min(minvalue_2,float(val["quantity"]))
#     minvalue_3 = min(minvalue_3,float(val["order"]))
#     minvalue_4 = min(minvalue_4,float(val["duration"]))
#     maxvalue_1 = max(maxvalue_1,float(val["price"]))
#     maxvalue_2 = max(maxvalue_2,float(val["quantity"]))
#     maxvalue_3 = max(maxvalue_3,float(val["duration"]))
#     maxvalue_4 = max(maxvalue_4,float(val["order"]))
#     if float(val["category"]) not in category:
#             category.append(float(val["category"]))



# minvalue_1 
# minvalue_2
# minvalue_3
# minvalue_4
# maxvalue_1
# maxvalue_2
# maxvalue_3
# maxvalue_4
# category


#------------------
#test here there are missing values  in the test (no uniqye numbers)

minvalue_1=999999999999999.999999999999 #price
minvalue_2=999999999999999.999999999999 #quantity
minvalue_3=999999999999999.999999999999 #order
minvalue_4=999999999999999.999999999999 #duration 
maxvalue_1=0 #price
maxvalue_2=0 #quantity
maxvalue_3=0 #order
maxvalue_4=0 #duration 
category = []
cat = np.zeros((33,), dtype=int)

import csv
import pickle 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
# Open file 
with open('pricing2.csv') as file_obj:
      
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    #next(reader_obj,None)
    # Iterate over each row in the csv 
    # file using reader object
    for row in reader_obj:
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = float(row[4])
        row[5] = int(row[5])
        minvalue_1 = min(minvalue_1,row[1])
        minvalue_2 = min(minvalue_2,row[2])
        minvalue_3 = min(minvalue_3,row[3])
        minvalue_4 = min(minvalue_4,row[4])
        maxvalue_1 = max(maxvalue_1,row[1])
        maxvalue_2 = max(maxvalue_2,row[2])
        maxvalue_3 = max(maxvalue_3,row[3])
        maxvalue_4 = max(maxvalue_4,row[4])
        if row[5] not in category:
            category.append(row[5])
    category = np.sort(category) #mabye only 30

# category = np.sort(category)
# len(category)
# s = pd.Series(category)
# a = pd.get_dummies(pd.Series(category),prefix="category")
# a.values.argmax(1)
    ##data archtecture 
    inputs = tf.keras.layers.Input(shape=(36, ), name = 'input') 
    hidden1 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden1')(inputs)
    hidden2 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden2')(hidden1)
    hidden3 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden3')(hidden2)
    output = tf.keras.layers.Dense(units=1, activation='linear', name='output')(hidden3)
    model = tf.keras.Model(inputs = inputs, outputs = output)
    model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))  


# category[1]

# row[5] == category[25]
len(cat)

training_times = []# initialize a list to store the training time for each iteration
start_time = time.time() # get the start time
initial_memory = psutil.Process().memory_info().rss # get the initial memory usage

df=open('Result file','w')
# Open file 
with open('pricing2_p2.csv') as file_obj:
      
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    #next(reader_obj,None)
    # Iterate over each row in the csv 
    # file using reader object

    i = 0
    for row in reader_obj:
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = float(row[4])
        row[5] = int(row[5])
        norm_1 = (row[1]/(maxvalue_1-minvalue_1)) #price
        norm_2 = (row[2]/(maxvalue_2-minvalue_2)) #quantity
        norm_3 = (row[3]/(maxvalue_3-minvalue_3)) #order
        norm_4 = (row[4]/(maxvalue_4-minvalue_4)) #duration
        cat = np.zeros((33,), dtype=int)
        index = row[5]
        cat[index]=1
#        X =np.array(np.column_stack((norm_1,norm_3,norm_4,cat)))
        X1 = np.append(cat,[norm_1,norm_3,norm_4])
        X = X1.reshape(1,-1)
        y = np.array(norm_2)
        y = y.reshape(1,-1)
        result = model.fit(x=X, y=y, batch_size = 1, epochs = 1)
        a = pd.DataFrame(result.history)
        df.write(str(a))
        df.write("\n")
        i = i+1
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
            
    current_memory = psutil.Process().memory_info().rss
    memory_usage = current_memory - initial_memory
    iteration_time = time.time() - start_time
    training_times.append(iteration_time)
    category = np.sort(category) #mabye only 30
    total_time = time.time() - start_time
    average_time = sum(training_times) / len(training_times)
df.close()


print(f"Memory usage: {memory_usage} bytes")
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average training time per iteration: {average_time:.2f} seconds")

#X = [[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        1.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.14444365, 4.61834412,
        0.15198618],[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        1.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.2837647642, 4.12308449817,
        0.984671648791],[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        1.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.3924861764, 4.293846257829,
        0.239847298742]]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

yhat = model.predict(x=X)  
#this is the line that we fit our test data to know the prediction. 

model.evaluate(x=X,y=y)