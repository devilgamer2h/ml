import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_graphviz
from IPython.display import Image

import pydotplus
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r"C:\Users\CAED31\Desktop\titanic.csv")

pd.set_option('display.max_columns',None)

data.head()

data.shape

data.info()

data.Survived.unique()

data.isnull().sum()

df= data.drop(['Cabin'] ,axis=1)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

df = df.drop(columns=['PassengerId', 'SibSp', 'Parch', 'Ticket',"Name"])

df.columns

df = pd.get_dummies(df, columns=['Sex', 'Embarked',"Pclass"], drop_first=True)

X=df.drop(columns=['Survived'])
y=df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier( random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

plt.figure(figsize=(12,8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Survived','UnSurvived'])
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:")
print(cm)

print("\n Classification Report:")
print(classification_report(y_test,y_pred))

