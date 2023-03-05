import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
import psutil

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

#import pickle  
# Open file 
with open('pricing_2.csv') as file_obj:
      
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

#timing and RAM


#model = tf.keras.models.load_model('my_model0.h5')
checkpoint_path = "training_0/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                 save_weights_only = True,
                                                 verbose = 1)

#This creates a single collection of tf checkpoint files that are updated at the end of each epoch
os.listdir(checkpoint_dir)

# csvfile = open("Result.csv",'w',newline='')
# df = csv.writer(csvfile)


#models = [] 

# Open file 
with open('pricing_2.csv') as file_obj:
    start_time = time.time() # get the start time

      
    # Create reader object by passing the file 
    # object to reader method
    reader_obj = csv.reader(file_obj)
    #next(reader_obj,None)
    # Iterate over each row in the csv 
    # file using reader object
    #for row in range(1,740000):#range(line start to skip, line end to skip+1)
          #next(reader_obj)
    i = 0
    for row in reader_obj:
        row[1] = float(row[1])
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = float(row[4])
        row[5] = int(row[5])
        norm_1 = ((row[1]-minvalue_1)/(maxvalue_1-minvalue_1)) #price
        norm_2 = ((row[2]-minvalue_2)/(maxvalue_2-minvalue_2)) #quantity
        norm_3 = ((row[3]-minvalue_3)/(maxvalue_3-minvalue_3)) #order
        norm_4 = ((row[4]-minvalue_4)/(maxvalue_4-minvalue_4)) #duration
        cat = np.zeros((33,), dtype=int)
        index = row[5]
        cat[index]=1
        X1 = np.append(cat,[norm_1,norm_3,norm_4])
        X = X1.reshape(1,-1)
        y = np.array(norm_2)
        y = y.reshape(1,-1)
        model.fit(x=X, y=y, batch_size = 1, epochs = 1, callbacks = [cp_callback]) #Pass callback to training 
        if i % 5000 ==0: 
            initial_memory = psutil.Process().memory_info().rss # get the initial memory usage
            result = model.fit(x=X, y=y, batch_size = 1, epochs = 1, callbacks = [cp_callback]) #Pass callback to training 

            # a = pd.DataFrame(result.history)
            # df.writerow(a.iloc[0])
            with open("Result0.txt",'a') as f:
                f.write(f"Processed {i} records")
                f.write(str(result.history['loss']))
                f.write(f"Memory usage: {psutil.Process().memory_info().rss - initial_memory} bytes\n")
                f.write('\n')
            model.save("my_model0.h5")
        i = i+1

        current_memory = psutil.Process().memory_info().rss
        memory_usage = current_memory - initial_memory
        category = np.sort(category) #mabye only 30


# model.save("my_model.h5")


#with open("model.pkl", "wb") as f: #this line should be change as the object above model_x x = 1,2,3
#    pickle.dump(models, f)  

f.close()
total_time = time.time() - start_time

# This line means load the training model first, Run this Chunk before making predictation
#with open("model.pkl", "rb") as f:
#    model = pickle.load(f)


#Print time and RAM 
print(f"Memory usage: {memory_usage} bytes")
print(f"Total training time: {total_time:.2f} seconds")
print(f"Average training time per iteration: {average_time:.2f} seconds")




#test data
with open('pricing_test.csv') as file_obj:
      
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
        cat = np.zeros((33,), dtype=int)
        index = row[5]
        cat[index]=1
#        X =np.array(np.column_stack((norm_1,norm_3,norm_4,cat)))
        X1 = np.append(cat,[row[1],row[3],row[4]])
        X = X1.reshape(1,-1)
        # y = np.array(norm_2)
        # y = y.reshape(1,-1)
        yhat = new_model.predict(x=X)

        with open("pricing_test.txt",'a') as f:
            f.write(f"yhat {i} : {yhat} \n")
        i = i+1




yhat = new_model.predict(x=X)  
#this is the line that we fit our test data to know the prediction. 

new_model.evaluate(x=X,y=y)


yhat=model.predict(x=X)

y_hat = yhat.reshape(-1)
#y.dtype

def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)

r_squared(y,y_hat)


import os
import tensorflow as tf
from tensorflow import keras


new_model = tf.keras.models.load_model('my_model2.h5')
new_model.summary()

