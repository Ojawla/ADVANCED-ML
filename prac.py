#missing data then duplicates then outliers if there are numrical hot encoding if there are categorical and then feature scaling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('penguins_size.csv')

#add maybe 1 or 2 pics to data understanding
print(df.head())
print(df.shape)
print(df.info())


#missing values make sure you take picof before adressing and thn after
print(df.isnull().sum())


#how to adress missing values median imputation for numerical column and most requent imputation for categrical
from sklearn.impute import SimpleImputer

# Numerical columns: median
num_imputer = SimpleImputer(strategy='median')

numerical = df.select_dtypes(include=np.number).columns
df[numerical] = num_imputer.fit_transform(df[numerical])

# Categorical columns: most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')

categorical = df.select_dtypes(include='object').columns
df[categorical] = cat_imputer.fit_transform(df[categorical])


print(df.isnull().sum())


#justification for median
#if there are missing vaues answer will be wrong
#The median imputation technique was selected to replace missing numerical values because it is less affected by outliers and extreme values compared to the mean. This helps maintain a representative value for the dataset."

#justification for most frequent
#Most frequent imputation was selected for categorical variables because it replaces missing values with the most commonly occurring category, preserving existing categories within the dataset.

#adressing dupicates
print(df.duplicated().sum())

#if there are duplicates
df = df.drop_duplicates()
#Duplicate records were removed to prevent the same observations from being counted multiple times. Retaining duplicates could introduce bias into the dataset, potentially affecting model training and leading to less reliable predictions


#outliers
# Check outliers using boxplot
# Check outliers in all numerical columns

numerical = df.select_dtypes(include=np.number).columns
#was no outlierbut the othe roe had outlier
for col in numerical:
    sns.boxplot(x=df[col])
    plt.title(f'Outliers in {col}')
    plt.show()

# One-Hot Encoding
# One-Hot Encoding

categorical=df.select_dtypes(include=['object', 'str']).columns
df = pd.get_dummies(df, columns=categorical, dtype=int)
#this the after so take before and after hot encoding
print(df.head())
print(df.info())
#justify
#One-Hot Encoding was performed to convert categorical variables into numerical values (0s and 1s), allowing the machine learning model to process categorical data. Each category is represented as a separate binary column.


#feature scaling not all modl need it
#tree based models dont need it eg random forestdecision trees


'''

data = {
    'Age': [21, 22, 23, 24, 25, 26, 27, 28, 29, 95],
    'Salary': [22000, 24000, 25000, 26000, 27000,
               28000, 29000, 30000, 31000, 200000],
    'Experience': [1, 2, 2, 3, 3, 4, 4, 5, 5, 40],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR',
                   'IT', 'Finance', 'HR', 'IT', 'Finance']
}


#example with outlier and justification
df = pd.DataFrame(data)
numerical = df.select_dtypes(include='number').columns
for col in numerical:
    sns.boxplot(x=df[col])
    plt.title(f'Outliers in {col}')
    plt.show()

# Firstly, boxplots were generated for all numerical features to visually investigate potential outliers. Identifying outliers is important because extreme values can distort the distribution of data and potentially negatively affect model performance, resulting in less reliable prediction.he visualisation identified potential outliers in the [insert column names] features, which were subsequently investigated using the IQR method to determine whether they fell outside the acceptable boundaries.
for col in numerical:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f'\nOutliers in {col}:')
    print(outliers)

#iqr method was selected because it identifies extreme values based on the spread of the data and is less sensitive to extreme observations than mean-based methods

#cap the outliers
for col in numerical:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

#why iqr capping
#The identified outliers were addressed using the IQR capping technique. This involves replacing values below the lower boundary or above the upper boundary with their respective IQR boundaries. Capping was selected instead of removing entire records because it retains observations within the dataset whilst reducing the influence of extreme values.


#Categorical encoding convert non numerical data into numerical values so ml model can understand and proces sthem
#hot encoding
'''

#for ths let practice clustering
#clustering prediciting what group customer belong to withouth predefined labels
#group similar customers but if it already ha slabels any you predict thats classification


#kmeans is my bseline
#fuzzy cmeans si my main



