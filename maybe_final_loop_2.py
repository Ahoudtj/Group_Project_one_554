

import os
import pandas as pd

chunks = pd.read_csv('pricing.csv', sep=',', chunksize=1)
if not os.path.exists('chunk_files'):
    os.makedirs('chunk_files')
results = []
for i, chunk in enumerate(chunks):
    # standardize the numerical columns (excluding category)
    for col in chunk.columns:
        if col not in ['category', 'sku']:
            chunk[col] = (chunk[col] - chunk[col].mean()) / chunk[col].std()
    # create dummy columns for the category column
    dummy_columns = pd.get_dummies(chunk['category'], prefix='category')
    # concatenate the dummy columns with the original dataframe
    chunk = pd.concat([chunk.drop('category', axis=1), dummy_columns], axis=1)
    # save the chunk to a CSV file
    filename = f'chunk_{i}.csv'
    chunk.to_csv(f'chunk_files/{filename}', index=False)
    # add the chunk to the results list
    results.append(chunk)

# concatenate all of the chunks into a single dataframe
price_train = pd.concat(pd.read_csv(f'chunk_files/{f}') for f in os.listdir('chunk_files'))
