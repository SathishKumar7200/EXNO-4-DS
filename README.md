# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
### STEP 1:
       Read the given Data.
### STEP 2:
       Clean the Data Set using Data Cleaning Process.
### STEP 3:
       Apply Feature Scaling for the feature in the data set.
### STEP 4:
       Apply Feature Selection for the feature in the data set.
### STEP 5:
       Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
from google.colab import drive
drive.mount('/content/drive')


![Screenshot 2024-12-14 215155](https://github.com/user-attachments/assets/068f5729-639b-4d08-a20e-a93f14490142)

ls drive/MyDrive/bmi.csv


![Screenshot 2024-12-14 215206](https://github.com/user-attachments/assets/f8fa9496-e63d-4b5e-b17b-d3fdade2845b)

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("drive/MyDrive/bmi.csv")

df.head()


![Screenshot 2024-12-14 215220](https://github.com/user-attachments/assets/6f3ab16d-6dd2-4aa1-ac2b-436721960929)

df.dropna()

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals


![Screenshot 2024-12-14 215231](https://github.com/user-attachments/assets/c133e052-f789-473e-9e76-bc99b7c4d32f)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)


![Screenshot 2024-12-14 215243](https://github.com/user-attachments/assets/ab662921-8ef0-4649-ad4f-8cf5cb91cdd9)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)


![Screenshot 2024-12-14 215256](https://github.com/user-attachments/assets/fd537d0a-9a1f-4957-b6d6-8e055f909bdb)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df


![Screenshot 2024-12-14 215306](https://github.com/user-attachments/assets/72fb8651-49b3-4dcb-8287-05892eff2f2f)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df


![Screenshot 2024-12-14 215315](https://github.com/user-attachments/assets/a1a5d343-2bd3-40b2-9ec1-829543e97283)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()


![Screenshot 2024-12-14 215326](https://github.com/user-attachments/assets/be3cb3f5-25b9-4e4a-935b-cfd8744ea336)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

df.columns


![Screenshot 2024-12-14 215337](https://github.com/user-attachments/assets/4a46536d-ac62-4a28-aa37-eb74dca0776d)

df.shape


![Screenshot 2024-12-14 215346](https://github.com/user-attachments/assets/ba76f553-7430-412d-b38e-8dad2d93f174)

x=df.drop('Survived',axis=1)
y=df['Survived']

df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df.columns


![Screenshot 2024-12-14 215358](https://github.com/user-attachments/assets/98a365a4-1781-4ce1-be54-e2806a9632ae)

df['Age'].isnull().sum()


![Screenshot 2024-12-14 215410](https://github.com/user-attachments/assets/94ff8937-94be-4130-81c0-de6655fb4815)

df['Age'].fillna(method='ffill')


![Screenshot 2024-12-14 215432](https://github.com/user-attachments/assets/813cf8b3-be76-41e6-a742-f7b0a82d431a)

df['Age']=df['Age'].fillna(method='ffill')
df['Age'].isnull().sum()


![Screenshot 2024-12-14 215445](https://github.com/user-attachments/assets/0e9d48bf-3c71-4eb9-b289-e6e8cc1da7fb)

data=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x


![Screenshot 2024-12-14 215456](https://github.com/user-attachments/assets/dc87cca9-0437-4bba-aa91-cc3619dff00a)

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data


![Screenshot 2024-12-14 215512](https://github.com/user-attachments/assets/40d68ce2-dd22-495f-a60b-df12e48db7b8)

for column in['Sex','Cabin','Embarked']:
   if x[column].dtype=='object':
             x[column]=x[column].astype('category').cat.codes
k=5
selector=SelectKBest(score_func=chi2,k=k)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 215526](https://github.com/user-attachments/assets/dc8a1bab-b0fb-4207-99fa-e861cd908aa5)

x.info()


![Screenshot 2024-12-14 215537](https://github.com/user-attachments/assets/bca1dc7c-80ef-44cd-b6cb-646f13da6071)

x=x.drop(["Sex","Cabin","Embarked"],axis=1)
x


![Screenshot 2024-12-14 215549](https://github.com/user-attachments/assets/4a77ff0b-5640-4925-8a48-aa1273d8ff49)

from sklearn.feature_selection import SelectKBest, f_regression
selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 215559](https://github.com/user-attachments/assets/34c40c8d-85a6-45cc-83fb-a486eb0464ae)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
X_new=selector.fit_transform(x,y)

selected_features_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 215610](https://github.com/user-attachments/assets/6e2bc545-1355-4f6f-b3af-ec35935b71e3)

from sklearn.feature_selection import SelectPercentile,chi2
selector=SelectPercentile(score_func=chi2,percentile=10)
x_new=selector.fit_transform(x,y)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 215621](https://github.com/user-attachments/assets/8f3256d5-a0cd-45c1-a909-7c1c55d5d149)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance = model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance > threshold]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 215631](https://github.com/user-attachments/assets/2f0851ab-69d5-4554-acc1-0054bb42bc10)

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")
df.columns


![Screenshot 2024-12-14 215640](https://github.com/user-attachments/assets/555061af-120f-4dac-a4c4-d57f651aa847)

df


![Screenshot 2024-12-14 215651](https://github.com/user-attachments/assets/28b68abd-5122-4423-85fc-b4bca51db5c6)

df.isnull().sum()


![Screenshot 2024-12-14 215753](https://github.com/user-attachments/assets/c6511aac-1843-4b07-b3f9-76430d36bfa6)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()


![Screenshot 2024-12-14 215806](https://github.com/user-attachments/assets/56bfa03a-8714-417d-9a25-85fd2d3b1e4d)

contigency_table=pd.crosstab(tips["sex"],tips["time"])
contigency_table


![Screenshot 2024-12-14 215818](https://github.com/user-attachments/assets/5e43b309-9bd3-4004-b133-7fdf0762c345)

chi2,p,_,_=chi2_contingency(contigency_table)
print(f"chi-Squared Statistic: {chi2}")
print(f"p-value: {p}")


![Screenshot 2024-12-14 215828](https://github.com/user-attachments/assets/d5c8f80c-8737-42bc-8cd5-dce42bc41967)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

      data={
      'Feature1':[1,2,3,4,5],
      'Feature2':['A','B','C','A','B'],
      'Feature3':[0,1,1,0,1],
      'Target':[0,1,1,0,1]
      }
      df=pd.DataFrame(data)
      x=df[['Feature1','Feature3']]
      y=df['Target']
      selector = SelectKBest(score_func=f_classif, k=2)
      selector.fit(x, y)
      selector_feature_indices=selector.get_support(indices=True)
      selected_features=x.columns[selector_feature_indices]
      print("Selected Features:")
      print(selected_features)
      print("selected_Features:")
      print(selected_features) # Assuming selected_features holds the desired value

      
![Screenshot 2024-12-14 215843](https://github.com/user-attachments/assets/0e4fc515-ba1b-4af1-959f-9bb9e3de2ed2)

```python
import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("/content/bmi.csv")

df.head()

df.dropna()

# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE

from sklearn.preprocessing import MinMaxScaler

#Perform minmax scaler

from sklearn.preprocessing import StandardScaler



```
       
# RESULT:
       The above code is excuted successfully.
