import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("HepatitisCdata.csv")

cols_means ={}

def fit(data):
    cols_with_na = data.isna().sum()[data.isna().sum()>0].index.tolist()
    for col in cols_with_na:
        cols_means[col] = data.groupby('Category')[col].transform('mean')

def transform(data):
    cols_with_na = data.isna().sum()[data.isna().sum()>0].index.tolist()
    for col in cols_with_na:
        mean_value = cols_means[col]
        data.loc[:,col].fillna(mean_value, inplace = True)

def prepocessing(data):
    fit(data)
    transform(data)
   
prepocessing(df)

y ={}
X = {}

def generate_train_test_data(data):
    data.drop('Unnamed: 0', axis =1 , inplace = True)
    y = data['Category'].replace({'0=Blood Donor':0, '0s=suspect Blood Donor':0 ,'2=Fibrosis':0,'3=Cirrhosis':0,'1=Hepatitis':1})
    data['Sex'].replace({'m':1,'f':0}, inplace =True)
    X = data.drop('Category', axis =1 )
    return y ,X

y, X = generate_train_test_data(df)

X_train, X_test, y_train, y_test  = train_test_split(X,y)

model = SVC()
prepocessing(X_train)
prepocessing(X_test)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
a = accuracy_score(y_pred,y_test)
print('accuracy score : ',a)
