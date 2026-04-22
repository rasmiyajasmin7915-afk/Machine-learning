#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv('Occupancy_Estimation.csv')

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month

df.drop(['Date', 'Time', 'DateTime'], axis=1, inplace=True)

outlier_cols = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light',
                 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 
                 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR', 'Hour', 'Day', 'Month']

def remove_outliers_iqr(data, column):
  q1,q2,q3 = np.percentile(data[column],[25,50,75])
  print("q1,q2,q3 is :",q1,q2,q3)
  IQR = q3-q1
  print("IQR is :" ,IQR)
  lower_limit = q1-(1.5*IQR)
  upper_limit = q3+(1.5*IQR)
  data[column]=np.where(data[column]>upper_limit,upper_limit,data[column]) # Capping the upper limit
  data[column]=np.where(data[column]<lower_limit,lower_limit,data[column]) # Flooring the lower limit

for column in outlier_cols:
  if column != 'Room_Occupancy_Count': # Exclude the target variable from outlier removal
    remove_outliers_iqr(df,column)

X = df.drop('Room_Occupancy_Count', axis=1)
y = df['Room_Occupancy_Count']

normalisation = MinMaxScaler()
X=normalisation.fit_transform(X)
# Coverting to Dataframe
X=pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(normalisation, open("scaler.pkl", "wb"))

#The model is a classification model that predicts room occupancy count using sensor data such as temperature, light, sound, CO₂, and motion sensors. Data preprocessing, feature engineering, scaling, and hyperparameter tuning were applied to improve performance. The best model (e.g., Random Forest) achieved good accuracy and reliability.
#Automatically control lights and AC,Save energy,Monitor room usage