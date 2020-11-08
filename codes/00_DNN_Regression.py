import pandas as pd
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

from helper_functions import *

# Read dataset
df = pd.read_csv('data/SeoulBikeData.csv',
                 parse_dates=['Date'],
                 encoding='cp1250')

# Date features
df['Date_Month'] = df['Date'].apply(lambda x: x.month)
df['Date_Day'] = df['Date'].apply(lambda x: x.day)
df['Date_Weekday'] = df['Date'].apply(lambda x: x.weekday())
df.drop(columns=["Date"], inplace=True)

# Get dummies
df = pd.get_dummies(df, drop_first=True,
                    columns=['Seasons', 'Holiday', 'Functioning Day'])

# Split features, labels
features = df.drop(columns=['Rented Bike Count']).values
labels = df['Rented Bike Count'].values

batch_size = 10
