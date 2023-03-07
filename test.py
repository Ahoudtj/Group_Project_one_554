import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
import psutil
import sys
import gc 

minvalue_1=999999999999999.999999999999 #price
minvalue_2=999999999999999.999999999999 #quantity
minvalue_3=999999999999999.999999999999 #order
minvalue_4=999999999999999.999999999999 #duration 
maxvalue_1=0 #price
maxvalue_2=0 #quantity
maxvalue_3=0 #order
maxvalue_4=0 #duration 
category = []

with open('pricing.csv', 'r') as file_obj: 
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    header = next(reader_obj) # read the header row
    if header[0].startswith('\ufeff'):
        header[0] = header[0][1:]
    # Iterate over each row in the csv 
    # file using reader object
    for row in reader_obj:
        # Skip row if it contains an empty string
        if '' in row:
            continue
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = float(row[4])
        row[5] = int(row[5])
        minvalue_1 = min(minvalue_1, row[1])
        minvalue_2 = min(minvalue_2, row[2])
        minvalue_3 = min(minvalue_3, row[3])
        minvalue_4 = min(minvalue_4, row[4])
        maxvalue_1 = max(maxvalue_1, row[1])
        maxvalue_2 = max(maxvalue_2, row[2])
        maxvalue_3 = max(maxvalue_3, row[3])
        maxvalue_4 = max(maxvalue_4, row[4])
        if row[5] not in category:
            category.append(row[5])
    category = np.sort(category)


##data archtecture 
inputs = tf.keras.layers.Input(shape=(36, ), name = 'input') 
hidden1 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=36, activation='sigmoid', name='hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation='linear', name='output')(hidden3)
model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))  


model = tf.keras.models.load_model('my_model_n.h5')

#model.save("my_model_n.h5")

with open('pricing.csv', 'r') as file_obj: 
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj, delimiter=',')
    header = next(reader_obj) # read the header row
    if header[0].startswith('\ufeff'):
        header[0] = header[0][1:]
    i = 275000
    for row in range(1,275000):#range(line start to skip, line end to skip+1)
    #next(gen)
        for row in reader_obj:
            row[1] = float(row[1]) #price
            row[2] = float(row[2]) #quantities
            row[3] = float(row[3]) #order
            row[4] = float(row[4]) #duration
            row[5] = int(row[5]) #category
            norm_1 = ((row[1]-minvalue_1)/(maxvalue_1-minvalue_1)) #price
            norm_2 = ((row[2]-minvalue_2)/(maxvalue_2-minvalue_2)) #quantity
            norm_3 = ((row[3]-minvalue_3)/(maxvalue_3-minvalue_3)) #order
            norm_4 = ((row[4]-minvalue_4)/(maxvalue_4-minvalue_4)) #duration
            cat = np.zeros((33,), dtype=int)
            index = row[5]
            cat[index]=1
            X1 = np.append(cat, [norm_1, norm_3, norm_4])
            X = X1.reshape(1, -1)
            y = np.array(norm_2)
            y = y.reshape(1, -1)
            if i % 5000 == 0:
                initial_memory = psutil.Process().memory_info().rss
                for x in ['i', 'file_obj', 'minvalue_1', 'maxvalue_1', 'minvalue_2', 'maxvalue_2', 'minvalue_3', 'maxvalue_3', 'minvalue_4', 'maxvalue_4', 'model']:
                    print(x,':',sys.getsizeof(vars()[x]))
            result = model.fit(x=X, y=y, batch_size=1, epochs=1) # Pass callback to training #train on batch 
            del X, y
            if i % 5000 == 0: #print odj size
                with open("Result_new.txt", 'a') as f:
                    f.write(f"Processed {i} records\n")
                    f.write(str(result.history['loss']) + "\n")
                    f.write(f"Memory usage: {psutil.Process().memory_info().rss - initial_memory} bytes\n")
#                   f.write(f"{sys.getsizeof(vars()[i])}\n")
                    f.write("\n")
                model.save("my_model_n.h5")
            else:
                del result
            i += 1
            gc.collect()
f.close()

