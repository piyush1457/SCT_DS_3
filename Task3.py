import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('bank.csv', delimiter=';')
print("First few rows of the dataset:")
print(df.head())
print("\nMissing values in each column:")
print(df.isnull().sum())
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=['y_yes'])
y = df_encoded['y_yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the Decision Tree model:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
