import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

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
X = df.drop(columns=['Rented Bike Count']).values.astype(np.float32)
X = normalize(X)
y = df['Rented Bike Count'].values.astype(np.float32)

# Split train, test
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3)
batch_size = 100
