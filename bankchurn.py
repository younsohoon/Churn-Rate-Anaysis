import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score 
import warnings
warnings.simplefilter(action='ignore')

train = pd.read_csv(".../Churn_Modelling.csv")

# Understanding data
train.info()
train.isna().sum()
train.CustomerId.unique()
train = train.dropna()

train = train.drop(columns=["CustomerId",'RowNumber','Surname'])
train = train.drop_duplicates()

categorical_columns = [var for var in train.columns if train[var].dtype == 'object']
numerical_columns = [var for var in train.columns if train[var].dtype != 'object']

# outliers 
train = train[(np.abs(stats.zscore(train[numerical_columns])) < 3).all(axis=1)]


# EDA

# correlations
train_corr = train.copy()
train_corr.columns

# processing categorical columns
label_encoder = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_corr[col] = label_encoder.fit_transform(train_corr[col])
    
train_corr = train_corr.corr()
plt.figure(figsize=(12,9))
sns.heatmap(data=train_corr,annot=True)


# Count number of categorical and numerical columns
# categorical 
fig, axes = plt.subplots(2,2,figsize=(12,10))
for idx, cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train,hue='Exited',ax=axes[row,col])

plt.subplots_adjust(hspace=0.5)

# numerical 
plt.figure(figsize=(12,9))
for i, col in enumerate(numerical_columns):
    plt.subplot(3,3,i+1)
    sns.histplot(data=train, x=col, bins=20, kde=True)


# Average age of each group by Gender
sns.boxplot(data=train, x=train.Exited, y=train.Age, hue='Gender')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# Average balance of each group by Gender
sns.boxplot(data=train, x=train.Exited, y=train.Balance, hue='Gender')
plt.legend(bbox_to_anchor=(1,1))
plt.show()

# Relation between Age, Balance and Churn
sns.scatterplot(data=train,x=train.Age,y=train.Balance,hue="Exited")
plt.legend(bbox_to_anchor=(1.2,0.8))



# Transforming data and creating model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

model = RandomForestClassifier(n_estimators=27)

# upsampling
exited = train[train.Exited==1]
no_exit = train[train.Exited==0]
upsampled_exited = resample(exited, replace=True, n_samples=len(no_exit))
data_model = pd.concat([upsampled_exited, no_exit])

data_model.Exited.value_counts()

for col in ['Geography', 'Gender']:
    data_model[col] = label_encoder.fit_transform(data_model[col])
    
X = data_model.drop(columns=['Exited'])
y = data_model['Exited']

# Splitting data, training models and evaluation
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=7)

model.fit(X_train,y_train)

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

y_predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print("Accuracy:", accuracy)
# print("Test F1 Score: ",f1_score(y_test,y_predicted))
print("Classification Report:")
print(classification_report(y_test, y_predicted))
print("Confusion Matrix on Test Data")
pd.crosstab(y_test, y_predicted, rownames=['True'], colnames=['Predicted'], margins=True)