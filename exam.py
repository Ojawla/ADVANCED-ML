#missing data then duplicates then outliers if there are numrical hot encoding if there are categorical and then feature scaling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


df = pd.read_csv("Mall_Customers.csv")
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))


#why kmeans justify

#Start with three numerical features if they exist in your file. Keep feature selection explicit; use only features you can justify
#These three features are chosen because they capture different aspects of customer behaviour and characteristics, allowing the clustering algorithm to identify meaningful customer segment
#we can see how age impact how incomeimapct how spendign scor eimpact
#added one categorical feature so cna try hot encoding
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = df[features].copy()

# One-Hot Encoding for one column if is multiple look on top
X = pd.get_dummies(X, columns=["Gender"], dtype=int)

print(X.head())
#justification
#One-Hot Encoding was performed on the Gender column because it contains categorical data. This converted the categories into separate binary numerical columns (0 and 1), making the feature suitable for use by the clustering algorithm.


#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))

#why scale
#Feature scaling was performed using StandardScaler to standardise the selected features to a common scale. This is important because K-Means uses Euclidean distance to measure similarities between customers. Without scaling, features with larger numerical ranges could disproportionately influence the clustering results

#tChoose K using evidence elbiw and shilouette
# :#this step we test different number of kk from 2 to 11 and created a elbow which will allow us pick by Helps identify where adding more clusters produces diminishing improvements in inertia.
#we look for the point where increasing the number of clusters produces much smaller reductions in inertia


#followin that we also did shilouete to ceck eh shilowtter the /...
## The Silhouette score was used alongside the Elbow method to evaluate how well-separated the clusters were.
# This helps identify a suitable number of clusters, as higher Silhouette scores generally indicate better-defined clusters.
ks = range(2, 11)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")
plt.tight_layout()
plt.show()
pd.DataFrame({"K": list(ks), "inertia": inertias,
"silhouette": silhouettes}).round(3)

# analyse graphs
#The optimal number of clusters was considered using both the Elbow Method and the Silhouette Score. The Elbow plot shows that the WCSS decreases as the number of clusters increases, with the curve beginning to flatten around K = 4, suggesting diminishing returns from adding further clusters. The Silhouette Score, however, is highest at K = 2, indicating that two clusters provide the strongest separation. Although K = 2 has the highest silhouette score, K = 4 could be considered if a more detailed segmentation is required.
#Silhouette plot How well-separated the clusters are The highest score

#fuzzy c mean advanced model so i select k using those 2
# did random state 42 so its reproducable


'''
if it dont work
#use terminal first: python -m pip install fuzzy-c-means
from fcmeans import FCM
fcm = FCM(n_clusters=k, random_state=42)
fcm.fit(X_scaled)
U = fcm.u # shape: n_observations × k
centres_scaled = fcm.centers
print("Membership matrix shape:", U.shape)
print("First five membership vectors:\n", np.round(U[:5], 3))
print("Row sums (should be about 1):", U.sum(axis=1)[:5].round(4))
df["fcm_cluster"] = U.argmax(axis=1) # provisional hard label only
df["fcm_max_membership"] = U.max(axis=1)
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]

#parameters exppalined
#fcmfit-fits the fcm modle using the scaeled data
#U = fcm.u Stores the membership values for each observation across all clusters.
#Unlike K-Means, observations can have different degrees of membership in multiple clusters.
# we also recordes the macimum mebership which is the highest membership fpr each observation
#also record membership gap:Measures the difference between the two highest membership values. A large gap indicates clearer cluster affiliation, while a small gap indicates greater overlap.
'''

#second way
#terminal python -m pip install scikit-fuzzy
import skfuzzy as fuzz

# Transpose data because skfuzzy expects features × observations
X_fcm = X_scaled.T

# Fit Fuzzy C-Means
centres_scaled, U, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_fcm,
    c=2, #so my k=4
    m=2,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)

# Transpose membership matrix back
U = U.T

# Assign provisional cluster labels
df["fcm_cluster"] = U.argmax(axis=1)

# Maximum membership
df["fcm_max_membership"] = U.max(axis=1)

# Membership gap
sorted_U = np.sort(U, axis=1)
df["fcm_membership_gap"] = sorted_U[:, -1] - sorted_U[:, -2]

print("Membership matrix shape:", U.shape)
print("First five membership vectors:\n", np.round(U[:5], 3))
print("Row sums:", U.sum(axis=1)[:5].round(4))


print("\nFuzzy C-Means customer results:")

print(df[[
    "fcm_cluster",
    "fcm_max_membership",
    "fcm_membership_gap"
]].head(10).round(3))
#c=4 no of clusters,
# m=2 fuzines sparameter   if is close to 1 observation stronglyto one cluster #m=2 because It is a commonly used standard value that allows some overlap between clusters without making memberships too evenly distributed.
#maxiteration=1000 It gives the model up to 1,000 attempts to improve the clustering before stopping.
#parameters exppalined
#fcmfit-fits the fcm modle using the scaeled data
#U = fcm.u Stores the membership values for each observation across all clusters.
#Unlike K-Means, observations can have different degrees of membership in multiple clusters.
# we also recordes the macimum mebership which is the highest membership fpr each observation
#also record membership gap:Measures the difference between the two highest membership values. A large gap indicates clearer cluster affiliation, while a small gap indicates greater overlap.


# analysing result
#Membership matrix shape: (200, 4) 200 rpw 4 column
#person 1 is in [[0.431 0.249 0.148 0.171]


comparison = []

for candidate_k in [3, 4, 5, 6]:

    # Fit Fuzzy C-Means using scikit-fuzzy
    centres_k, U_k, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_scaled.T,
        c=candidate_k,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=42
    )

    # Transpose membership matrix back
    U_k = U_k.T

    # Calculate maximum membership
    max_membership = U_k.max(axis=1)

    # Calculate membership gap
    sorted_memberships = np.sort(U_k, axis=1)
    gap = sorted_memberships[:, -1] - sorted_memberships[:, -2]

    # Store comparison results
    comparison.append({
        "K": candidate_k,
        "mean_max_membership": max_membership.mean(),
        "median_max_membership": np.median(max_membership),
        "mean_top_two_gap": gap.mean()
    })

# Display results
comparison_df = pd.DataFrame(comparison).round(3)

print("\nFCM comparison for different numbers of clusters:")
print(comparison_df)

#Mean max membership: Average of the highest membership value assigned to each observation. Higher means observations belong more strongly to one cluster.
#Median max membership: The median of the highest membership values. Higher means the typical observation has a clearer cluster assignment.
#Mean top-two gap: Average difference between the highest and second-highest membership values. Higher means less ambiguity between the two most likely clusters.

# would choose K = 3 because:
#It has the highest mean maximum membership (0.653), meaning observations have stronger membership in their assigned clusters.
#It has the highest median maximum membership (0.630), suggesting that cluster assignments are generally clearer.
#however thr mean top two gap was highest with 6 whivh showed that 6 achieves a slightly higher separation between the two most likely cluster memberships

#I selected K = 3 because it provides the strongest overall membership certainty, although K = 6 achieves a slightly higher separation between the two most likely cluster membership

# waning
#Interpret carefully:
# mean maximum membership and the top-two gap describe the sharpness of the fuzzy partition, not model accuracy. Different K values also change the number and meaning of the segments.
# they dont tell you how accurate it is

#can copy this
'''
The mean maximum membership and mean top-two gap were used to assess the sharpness of the fuzzy partition 
rather than the accuracy of the clustering solution. K = 3 achieved the highest mean and median maximum membership, suggesting clearer membership assignments. However, K = 6 achieved the highest top-two gap'''

'''
Higher mean maximum membership: Observations have stronger membership in their most likely cluster.

Higher top-two gap: There is a bigger difference between the first and second cluster membership, meaning less ambiguity.

Lower values: Membership assignments tend to be less clear, with more uncertainty about which cluster an observation belongs to.'''


plot_df = df.copy()
plt.figure(figsize=(7, 5))
sns.scatterplot(data=plot_df, x=features[1], y=features[2],
hue="fcm_cluster", palette="tab10")
plt.title("FCM: provisional labels by maximum membership")
plt.tight_layout()
plt.show()
# Membership sharpness
plt.figure(figsize=(7, 4))
sns.histplot(df["fcm_max_membership"], bins=20, kde=True)
plt.xlabel("Maximum FCM membership")
plt.title("How sharply are customers assigned?")
plt.tight_layout()
plt.show()

#graph interpretation
'''The histogram illustrates the distribution of maximum Fuzzy C-Means (FCM) membership values across customers. Most customers have maximum membership scores between approximately 0.68 and 0.90, with the distribution peaking around 0.75–0.78. This indicates that the majority of customers have a relatively strong association with one particular cluster, although their assignments are not entirely definitive. A smaller proportion of customers exhibit membership values above 0.90, suggesting highly confident cluster assignments, while customers with lower membership values may possess characteristics that overlap across multiple clusters'''


