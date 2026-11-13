
import pandas as pd
from sklearn.datasets import make_classification


#es! Exactly. For standard SMOTE, your input features (X) must be numerical.
# Create an imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)

# Convert into DataFrame
X = pd.DataFrame(X, columns=[
    'age', 'income', 'account_balance',
    'transactions', 'customer_activity'
])

y = pd.Series(y, name='churn')

# Check class distribution
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)


# 1. Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#nlp dont use smite


# 2. Apply SMOTE to training data only
from imblearn.over_sampling import SMOTE

# Check before SMOTE
print("Before SMOTE:")
print(y_train.value_counts())

X_train, y_train = SMOTE(random_state=42).fit_resample(
    X_train, y_train
)


# 3. Train Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# Check after SMOTE
print("After SMOTE:")
print(y_train.value_counts())


# 4. Make predictions using test data
y_pred = model.predict(X_test)


# 5. Evaluate the model
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


#use if is lik85% to

# hpow to label encode
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


or this way
y = y.map({
    'No Cancer': 0,
    'Cancer': 1
})
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split()

#if you use smote
from imblearn.over_sampling import SMOTE

print("Before SMOTE:")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train).value_counts())

#train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)

model.fit(X_train, y_train)

#make predictions
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
