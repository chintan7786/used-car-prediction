
'''

*******************************

IMPORTING LIBS

*******************************

'''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import r2_score
import re
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
%matplotlib inline



'''

*******************************

IMPORTING DATASET

*******************************

'''
df_train = pd.read_csv("train-data.csv")
df_test = pd.read_csv("test-data.csv")



'''

*******************************

DEALING WITH MISSING VALUES AND CLEANING DATASET

*******************************

'''
df_train.isna().sum()
pert_data_missing = (df_train.isnull().sum() / len(df_train)) * 100
df_test.isna().sum()

# renaming cols of train and test dataset
df_train = df_train.rename(columns = {'Unnamed: 0': 'id'})
df_test = df_test.rename(columns = {'Unnamed: 0': 'id'})
df_train.isna().sum() 


''' Seats Column '''
# seats col for Traning dataset
df_train.groupby('Seats')['id'].nunique()
seats_NAN = df_train.loc[df_train.Seats.isnull()]
unique_seat_values = collections.Counter(df_train['Seats'])
unique_seat_values.most_common()
df_train['Seats'].nunique()
df_train['Seats'].unique()
df_train["Seats"].fillna(value = 5.0, inplace=True)
df_train.Seats[df_train.Seats == 0.0] = 5.0
df_train.isna().sum()

# Seats colg for Testing dataset
df_test.groupby('Seats')['id'].nunique()
df_test['Seats'].nunique()
df_test['Seats'].unique()
df_test["Seats"].fillna(value = 5.0, inplace=True)
df_test.Seats[df_test.Seats == 0.0] = 5.0
df_test.isna().sum()


''' Mileage column '''
# Mileage colin training dataset
df_train.groupby('Mileage')['id'].nunique()
df_train['Mileage'].nunique()
df_train.Mileage[df_train.Mileage == '0.0 kmpl'] = np.nan
mileage_NAN = df_train.loc[df_train.Mileage.isnull()]
df_train['Mileage'] = df_train['Mileage'].apply(lambda x: 
                                                re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', 
                                                       r'\1', str(x)))
df_train['Mileage'] = df_train['Mileage'].astype(float)
unique_mil_rows = collections.Counter(df_train['Mileage'])
unique_mil_rows.most_common()
df_train['Mileage'].mode()
df_train['Mileage'].fillna(value = 17.0, inplace = True)
df_train.isna().sum()

# Mileage Col in Testing dataset
df_test.Mileage[df_test.Mileage == '0.0 kmpl'] = np.nan
df_test['Mileage'] = df_test['Mileage'].apply(lambda x: 
                                                re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', 
                                                       r'\1', str(x)))
df_test['Mileage'] = df_test['Mileage'].astype(float)
df_test['Mileage'].mode()
df_test['Mileage'].fillna(value = 17.0, inplace = True)
df_test.isna().sum()


''' Engine Column '''
# Engine col in Traning dataset
df_train.groupby('Engine')['id'].nunique()
engine_NAN = df_train.loc[df_train.Engine.isnull()]
unique_Engine_values = collections.Counter(df_train['Engine'])
unique_Engine_values.most_common()
df_train['Engine'] = df_train['Engine'].apply(lambda x: 
                                                re.sub(r'(\d+)\s(CC)', 
                                                       r'\1', str(x)))
df_train['Engine'] = df_train['Engine'].astype(float)
df_train['Engine'].mode()
df_train['Engine'].fillna(value = 1197.0, inplace = True)
df_train.isna().sum()

# Engine col in testing dataset
df_test.groupby('Engine')['id'].nunique()
df_test['Engine'] = df_test['Engine'].apply(lambda x: 
                                                re.sub(r'(\d+)\s(CC)', 
                                                       r'\1', str(x)))
df_test['Engine'] = df_test['Engine'].astype(float)
df_test['Engine'].mode()
df_test['Engine'].fillna(value = 1197.0, inplace = True)
df_test.isna().sum()


''' Power Column '''
# Power col in training dataset
df_train['Power'] = df_train['Power'].str.split(' ').str[0]    
df_train.Power[df_train.Power == 'null'] = np.NaN
df_train['Power'].isnull().sum()
unique_power_values = collections.Counter(df_train['Power'])
unique_power_values.most_common()
power_NAN = df_train.loc[df_train.Power.isnull()]
df_train['Power'] = df_train['Power'].astype(float)
df_train['Power'].mode()
df_train['Power'].fillna(value = 74, inplace = True)
df_train.isna().sum()

# Power col in Testing dataset
df_test['Power'] = df_test['Power'].str.split(' ').str[0]
df_test.Power[df_test.Power == 'null'] = np.NaN
df_test['Power'].isnull().sum()
df_test['Power'] = df_test['Power'].astype(float)
df_test['Power'].mode()
df_test['Power'].fillna(value = 74, inplace = True)
df_test.isna().sum()


''' Name Column '''
# Name col Training dataset
df_train['Name'] = df_train['Name'].str.split(' ').str[0]
df_train.groupby('Name')['id'].nunique()
df_train.Name[df_train.Name == 'Isuzu'] = 'ISUZU'

# Name col Testing dataset
df_test['Name'] = df_train['Name'].str.split(' ').str[0]
df_test.groupby('Name')['id'].nunique()


# deleting cols which are not required
del df_train['New_Price']
del df_test['New_Price']
df_train.isna().sum()
dataset = df_train.copy()


# checking all the unique values in Column
df_train.groupby('Fuel_Type')['id'].nunique()
df_train.groupby('Owner_Type')['id'].nunique()
df_train.groupby('Transmission')['id'].nunique()
dataset.groupby('Location')['id'].nunique()
df_train.groupby('Year')['id'].nunique()
dataset.groupby('Mileage')['id'].nunique()

# id column is not required
del df_train['id']
del df_test['id']

# checking data type of column
df_train.dtypes
df_test.dtypes

# coverting int to float
df_train['Year'] = df_train['Year'].astype(float)
df_train['Kilometers_Driven'] = df_train['Kilometers_Driven'].astype(float)
df_test['Year'] = df_test['Year'].astype(float)
df_test['Kilometers_Driven'] = df_test['Kilometers_Driven'].astype(float)


# for improving the accuracy
df_train['Price_log'] = np.log1p(df_train['Price'].values)
del df_train['Price']



'''

*******************************

PRICE PREDICTION

*******************************

'''
# getting rid of categorical data
df_train = pd.get_dummies(df_train, drop_first = True)

# Splitting the Dataset
X = df_train.drop(columns = ['Price_log'], axis = 1)
y = df_train.iloc[:, 6].values

# spliting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)


# fitting the data to Linear Regression
regressor_1 = LinearRegression()
regressor_1.fit(X_train, y_train)

# prediction
y_pred_1 = regressor_1.predict(X_test)

# prediction score
regressor_1.score(X_test,y_test)
r2_score(y_pred_1, y_test)


# fitting the data to Random Forest Tree Regression
regressor_2 = RandomForestRegressor(random_state = 0)
regressor_2.fit(X_train, y_train)

# prediction
y_pred_2 = regressor_2.predict(X_test)

# prediction score
regressor_2.score(X_test,y_test)
r2_score(y_pred_2, y_test)


#Fitting the data to Decision Tree Algorithm
regressor_3 = DecisionTreeRegressor(random_state = 0)
regressor_3.fit(X_train, y_train)

# prediction
y_pred_3 = regressor_3.predict(X_test)

# prediction score
regressor_3.score(X_test, y_test)
r2_score(y_pred_3,y_test)


# ridge = 906; lasso = 898


'''

*******************************

VISUALIZING THE DATASET

*******************************

'''
plt.style.use('ggplot')
plt.style.available
colors = ['#FF8C73','#66b3ff','#99ff99','#CA8BCA', '#FFB973', '#89DF38', 
          '#8BA4CA', '#ffcc99', '#72A047', '#3052AF', '#FFC4C4']

# Bar plot Year wise count
plt.figure(figsize = (10,8))
bar1 = sns.countplot(dataset['Year'])
bar1.set_xticklabels(bar1.get_xticklabels(), rotation = 90, ha = 'right')
plt.title('Count year wise', size = 24)
plt.xlabel('Year', size = 18)
plt.ylabel('Count', size = 18)
plt.show()

# Bar plot Fuel_type count
plt.figure(figsize = (7,7))
sns.countplot(dataset['Fuel_Type'])
plt.xticks(dataset['Fuel_Type'])
plt.title('Types of Fuel and count', size = 24)
plt.tight_layout()
plt.show()
    
# pie plot by Location
plt.pie(dataset['Location'].value_counts(), startangle = 90, 
        autopct = '%1.1f%%', colors = colors, 
        labels = dataset['Location'].unique())
centre_circle = plt.Circle((0,0),0.80,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.tight_layout()
plt.show()

# bar plot for Transmission
plt.figure(figsize = (7,7))
sns.countplot(dataset['Transmission'])
plt.xticks(dataset['Transmission'])
plt.title('Types of transmission', size = 24)
plt.tight_layout()
plt.show()








