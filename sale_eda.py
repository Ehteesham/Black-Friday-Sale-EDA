import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Black Friday Sale 
# Agenda is to  clean the data and prepare the data for model training

# importing the data set
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
# print(df_test.shape)

pd.set_option('display.max_columns', None)

df_head = df_train.head()
# print(df_head)

df_head1 = df_test.head()
# print(df_head1)

# columns of the train , test data set
# print(df_test.columns)
# print(df_train.columns)


# concatinate the both data set
df = pd.concat([df_train, df_test]).drop_duplicates()
# print(df_new.head())

# print(df.info())

# print(df.describe())

# delete the column which is not important
df.drop(['User_ID'], axis = 1, inplace=True)
# print(df.head())

null_value = df.isnull().sum()
# print(null_value)


# handling categorical data
df['Gender'] = pd.get_dummies(df['Gender'], drop_first = True)
# df['Gender'] = df['Gender'].map({'F':0, 'M':1})
# print(df.head())


# handle age
# to find the unique value in the data set
# print(df['Age'].unique())

df['Age'] = df['Age'].map({'0-17':1, '18-25':2, '26-35':3,'36-45':4, '46-50':5, '51-55':6, '55+':7})
# print(df.head(7))

# handling the City_Category
df_city = pd.get_dummies(df['City_Category'], drop_first = True)
# print(df_city)

df = pd.concat([df,df_city], axis=1)
# print(df.head(7))
df.drop('City_Category', axis = 1, inplace = True)
# print(df)

# missing value
# print(df.isnull().sum())

# focus on replacing the missing value
pc2_unique = df['Product_Category_2'].unique()
# print(pc2_unique)

pc2_val_count = df['Product_Category_2'].value_counts()
# print(pc2_val_count)

# In a discrete value or categorical value it;s always better to take mode for missing values

# Replace missing value with mode
df['Product_Category_2'] = df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])
# print(df['Product_Category_2'].isnull().sum())

# now for Product_Category_3

df['Product_Category_3'] = df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])
# print(df['Product_Category_3'].isnull().sum())

# print(df.columns)

df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+','')
# print(df.head())

# print(df.info())

# convert object into integers
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(np.int64)
# print(df.info())


df['B'] = df['B'].astype(np.int64)
df['C'] = df['C'].astype(np.int64)

# sns.barplot(data=df,x = 'Age',y ='Purchase', hue = 'Gender')
# plt.show()

# Purchasing of men is high then women
# sns.barplot(data=df,x = 'Product_Category_1',y ='Purchase', hue = 'Gender')
# sns.barplot(data=df,x = 'Product_Category_2',y ='Purchase', hue = 'Gender')
# sns.barplot(data=df,x = 'Product_Category_3',y ='Purchase', hue = 'Gender')
# plt.show()

df_test=df[df['Purchase'].isnull()]
# print(df_test.head())

df_train = df[~df['Purchase'].isnull()]  #this will give you the value which is not null 
# print(df_train.head())

# train test split
X = df_train.drop('Purchase', axis = 1)
print(X.head())

y = df_train['Purchase']
# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# feature scaling
X_train.drop('Product_ID', axis = 1, inplace= True)
X_test.drop('Product_ID', axis = 1, inplace= True)
sc = StandardScaler()
sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train your model