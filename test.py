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
# Open file 
with open('pricing1.csv') as file_obj:
      
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    next(reader_obj,None)
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
    category = np.sort(category)

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


#range(start,end) end 20 print to 19
# category[1]

# row[5] == category[25]





csvfile = open("Result.csv",'w',newline='')
df = csv.writer(csvfile)


models = [] # change this line before the training the model ex models_x  x = 1,2,3,4

# Open file 
with open('/Users/emmachang/Desktop/Class/BZAN554_DP/pricing_test copy.csv') as file_obj:
      
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    #next(reader_obj,None)
    # Iterate over each row in the csv 
    # file using reader object
    # for i in range() #range(line start to skip, line end to skip+1)
    #     next(reader_obj)
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
        df.writerow(a.iloc[0])
        i = i+1
        models.append(model) # this line should be change as the object above

with open("model.pkl", "wb") as f: #this line should be change as the object above model_x x = 1,2,3
    pickle.dump(models, f)   #this line should be change as the object above models_x x =1,2,3


csvfile.close()

# This line means load the training model first, Run this Chunk before making predictation
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


#



yhat = model.predict(x=X)  
#this is the line that we fit our test data to know the prediction. 

model.evaluate(x=X,y=y)