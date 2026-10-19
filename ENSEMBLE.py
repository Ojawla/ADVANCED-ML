#Ensemble=sing multiple machine-learning models together to make one better prediction.
# 3 main typres boosting bagging voting
#improve accuracy could reduce indivual error an dbias


import numpy as np
import pandas as pd
# Visualisation
from matplotlib import pyplot as plt
# Datasets
from sklearn.datasets import load_breast_cancer, make_classification
# Data splitting and validation
from sklearn.model_selection import (
train_test_split,
cross_val_score,
RepeatedStratifiedKFold
)
# Pre-processing
from sklearn.preprocessing import LabelEncoder
# Evaluation
from sklearn.metrics import accuracy_score, f1_score
# Base learners
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Ensemble methods
from sklearn.ensemble import (
AdaBoostClassifier,
RandomForestClassifier,
BaggingClassifier,
VotingClassifier
)

#Adaboost (ensemble prediction model tat combines  multiple weak learner to create a stong prediction model)
#stats with a simple model call weak learner check what that model got wong it give sthe wrong classifier more attention and then keep building several models sequentialluy combines the answers to make one stronger model
#AdaBoost builds models sequentially, with later learners focusing more on observations that previous learners struggled to classify

# Load dataset
breast_cancer = load_breast_cancer()
# Create feature DataFrame
X = pd.DataFrame(
   breast_cancer.data,
    columns=breast_cancer.feature_names
)
# Create categorical target
y = pd.Categorical.from_codes(
breast_cancer.target,
breast_cancer.target_names
)
# Convert target labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Dataset shape:", X.shape)
print("Target classes:", encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.25,
random_state=1,# Rando state 1 ensures the split is the same each time make the result consstent and easier to compare
stratify=y # ensures traning and testing have similar split of classes  so amount of boys in train is same in test can lead to misleading results if not proportionate
)
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])

#adaboost "Create an AdaBoost model that uses 200 very simple decision trees, where each new tree focuses on the mistakes made by the previous trees
boosting_model = AdaBoostClassifier(
estimator=DecisionTreeClassifier(
max_depth=1,
random_state=1 #Make each decision tree very simple one level split
),
n_estimators=200, #build 200 of the small tree so each tree focuses more on mistake of the next
random_state=1
)
boosting_model.fit(X_train, y_train)
boost_predictions = boosting_model.predict(X_test)
boost_accuracy = accuracy_score(
y_test,
boost_predictions
)
boost_f1 = f1_score(
y_test,
boost_predictions
)
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")

#F1 is useful because it considers both precision and recall, not just how many predictions were correc
#decision stump as the weak learner. A decision stump is simply a decision tree with:max_depth=1
#Why might combining 200 very simple decision trees produce a stronger model than using one decision stump? Combining 200 simple trees allows AdaBoost to correct mistakes made by previous trees.


#2 Random forest ( bagging)
#bagging models lots of trees built independently using slightly different samples(bootstrap sampling allows repeats) of data and preditcions are combined
#creating different training samples, training separate trees, and combining their predictions
#whiles boosting trees built one after another trees focusing on previous mistakes

bagging_model = RandomForestClassifier(
n_estimators=200,
max_depth=1,
random_state=1,
n_jobs=-1
)
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(
y_test,
bagging_predictions
)
bagging_f1 = f1_score(
y_test,
bagging_predictions
)
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

#fl is precison tp Of everything the model predicted as YES, 70% was actually YES.
#If recall = 30%, it means:The model only found 30% of all the cases that were actually YES.

#model Comparison
#Adaboost accuracy 0.96 and f1 score =0.94( strong balnce between precision and recall did good finding yes cases recall and avoiding false yess precision
#Adaboost perform better
#very minimal difference
#Random forest can be trained independently because each tree learns from its own random sample of the data.
#random forest is easier to parallerise because its tree are seperate
#Decision trees are commonly used in bagging because they can be high-variance learners


#classification dataset
X_knn, y_knn = make_classification(
n_samples=1000,
n_features=20,
n_informative=15,
n_redundant=5,
random_state=5
)
print("Dataset shape:", X_knn.shape)

#bagging( take dataset make several random models, train each model seperate, each model makes prediction, combines predicition)
#10 knn model each model predicts combine the predictio

#combine many knn prediciton to get 1
knn_bagging = BaggingClassifier(
estimator=KNeighborsClassifier(),
n_estimators=10,
random_state=1,
n_jobs=-1
)

#Dataset → 10 KNN models → combine their predictions → final prediction
#stratified cross validation means test the model using different train test split while keeping the proportion class imilar .It gives you more confidence that the model will work well on new, unseen data.
#High mean accuracy + low standard deviation = good, consistent model.Standard Deviation → how much the performance changes between tests.
#good when dealling with imbalanced dataset or small sample size becasue withouth stratificiation ensures that each fold preserve the exact same percentage of samples for each class

#repeateevstratfied cross validation
cv = RepeatedStratifiedKFold(
n_splits=10,
n_repeats=3,
random_state=1
)
knn_scores = cross_val_score(
knn_bagging,
X_knn,
y_knn,
scoring="accuracy",
cv=cv,
n_jobs=-1,
error_score="raise"
)
print(f"Mean Accuracy: {knn_scores.mean():.3f}")
print(f"Standard Deviation: {knn_scores.std():.3f}")

#mean=0.884
#sd=0.037
#0-0.02(low variation conistent) 0.02 to 0.05( moderate ariation 0.05+ high variation incosistent
#mean 0.9 t0 1 good below 0.7 bad mean is how accurate the model predicition are if tested 30 time each wil have difierent accuracy so average
# mean 88% good, it was 88% correct on average
#Mean = how accurate the model is on average across the tests.
#SD: tells you how much the accuracy changes from test to test.

#summary mean on average across all test ho accurate are the prediction cause each test has each predicition so average and sd means how far apart are thos eprediction
#i wont want one ot predict 90% accurate other 20%

#discussion
#Why might KNN benefit from bagging?: because bagging use smultiple knn model not one combining the predicitions can make the final more reliable and less affected by random changes in the data
#Why might using multiple KNN models be more robust than using a single KNN model?
#One KNN model might make a wrong prediction because of the particular data it sees. Multiple KNN models vote together, so one model's mistake is less likely to affect the final answer.
#What does the standard deviation tell us about model stability?
#it tells us how much the model accuracy changes between test low sd=consistent and high sd=

#Soft voting ensemble
#Soft voting combines the probabilities/confidence scores from several models.
#The class with the highest combined probability becomes the final prediction
#modela cat 70% dog 30% model b cat 90% dog -10% the condidence for cat higher so cat is the predciiton

#create dataset
X_svm, y_svm = make_classification(
n_samples=1000,
n_features=20,
n_informative=15,
n_redundant=5,
random_state=2)
def create_soft_voting_model():
    models = [
        ("svm1", SVC(
            probability=True,
            kernel="poly",
            degree=1,
            random_state=1
        )),
        ("svm2", SVC(
            probability=True,
            kernel="poly",
            degree=2,
            random_state=1
        )),
        ("svm3", SVC(
            probability=True,
            kernel="poly",
            degree=3,
            random_state=1
        )),
        ("svm4", SVC(
            probability=True,
            kernel="poly",
            degree=4,
            random_state=1
        )),
        ("svm5", SVC(
            probability=True,
            kernel="poly",
            degree=5,
            random_state=1
        ))
    ]

    ensemble = VotingClassifier(
        estimators=models,
        voting="soft"
    )

    return ensemble


# Step 3 — Create models for comparison

def create_soft_voting_models():
    models = {
        "SVM-1": SVC(
            probability=True,
            kernel="poly",
            degree=1,
            random_state=1
        ),
        "SVM-2": SVC(
            probability=True,
            kernel="poly",
            degree=2,
            random_state=1
        ),
        "SVM-3": SVC(
            probability=True,
            kernel="poly",
            degree=3,
            random_state=1
        ),
        "SVM-4": SVC(
            probability=True,
            kernel="poly",
            degree=4,
            random_state=1
        ),
        "SVM-5": SVC(
            probability=True,
            kernel="poly",
            degree=5,
            random_state=1
        ),
        "Soft Voting": create_soft_voting_model()
    }

    return models


# Step 4 — Evaluate using repeated cross-validation

models = create_soft_voting_models()
results = []
names = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )

    results.append(scores)
    names.append(name)

    print(
        f"{name:12s} "
        f"Mean = {scores.mean():.3f} | "
        f"Std = {scores.std():.3f}"
    )


# Step 5 — Visualise the results

plt.figure(figsize=(10, 5))

plt.boxplot(
    results,
    tick_labels=names,
    showmeans=True
)

plt.ylabel("Accuracy")
plt.title("Soft Voting vs Individual SVM Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# ADDED CAUSE ONLY ONE BOX PLOT SHOWS ASK TEACHER WHY NOT ALL

print(
    f"{name:12s} "
    f"Mean Accuracy = {scores.mean():.3f} | "
    f"Std = {scores.std():.3f}"
)
"""
def create_soft_voting_model():
    models = [
    ("svm1", SVC(
        probability=True,
        kernel="poly",
        degree=1,
        random_state=1
    )),
    ("svm2", SVC(
        probability=True,
        kernel="poly",
        degree=2,
        random_state=1
    )),
    ("svm3", SVC(
        probability=True,
        kernel="poly",
        degree=3,
        random_state=1
    )),
    ("svm4", SVC(
        probability=True,
        kernel="poly",
        degree=4,
        random_state=1
    )),
    ("svm5", SVC(
        probability=True,
        kernel="poly",
        degree=5,
        random_state=1
    ))
    ]
    ensemble = VotingClassifier(
    estimators=models,
    voting="soft"
)

    return ensemble
#Step 3 — Create models for comparison

def create_soft_voting_models():
    models = {
    "SVM-1": SVC(
        probability=True,
        kernel="poly",
        degree=1,
        random_state=1
    ),
    "SVM-2": SVC(
        probability=True,
        kernel="poly",
        degree=2,
        random_state=1
    ),
    "SVM-3": SVC(
        probability=True,
        kernel="poly",
        degree=3,
        random_state=1
    ),
    "SVM-4": SVC(
        probability=True,
        kernel="poly",
        degree=4,
        random_state=1
    ),
    "SVM-5": SVC(
        probability=True,
        kernel="poly",
        degree=5,
        random_state=1
    ),
    "Soft Voting": create_soft_voting_model()
    }
    return models

#Step 4 — Evaluate using repeated cross-validation
models = create_soft_voting_models()
results = []
names = []
for name, model in models.items():
    scores = cross_val_score(
    model,
    X_svm,
    y_svm,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    error_score="raise"
)
results.append(scores)
names.append(name)
print(
f"{name:12s} "
    f"Mean = {scores.mean():.3f} | "
    f"Std = {scores.std():.3f}"
)
#Step 5 — Visualise the results
plt.figure(figsize=(10, 5))
plt.boxplot(
results,
tick_labels=names,
showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Soft Voting vs Individual SVM Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#ADDED CAUSE ONLY ONE BOX PLOT SHOWS ASK TEACHER WHY NOT ALL
print(
    f"{name:12s} "
    f"Mean Accuracy = {scores.mean():.3f} | "
    f"Std = {scores.std():.3f}"
)
"""
#Does Soft Voting outperform every individual SVM?
#Yes. Soft Voting has a median accuracy of about 0.93, which is higher than the median of every individual SVM. It also reaches the highest overall accuracy, around 0.96
#What does the spread of each box tell you? The box represents the middle 50% of accuracy values (the interquartile range, or IQR
#Smaller box: results are more consistent and less variable.
#Larger box: results vary more between runs/data samples.
#The whiskers show the broader range of typical results.

#The spread of each box shows how much the model's accuracy varies across the different cross-validation folds.
#Combining models with different polynomial degrees can improve performance because each SVM can learn different patterns and relationships in the data.
#Lower-degree polynomials (such as degree 1 or 2) learn simpler patterns.
#Higher-degree polynomials (such as degree 4 or 5) can learn more complex patterns.

#Hard voting
#Hard voting: models vote for a class → most votes wins.
#Soft voting: models give probabilities → probabilities are combined → highest probability win


#Step 1 — Create the KNN voting ensemble Odd values reduce the possibility of ties.
X_svm, y_svm = make_classification(
n_samples=1000,
n_features=20,
n_informative=15,
n_redundant=5,
random_state=2)
def create_hard_voting_model():
    models = [
        ("knn1", KNeighborsClassifier(n_neighbors=1)),
        ("knn3", KNeighborsClassifier(n_neighbors=3)),
        ("knn5", KNeighborsClassifier(n_neighbors=5)),
        ("knn7", KNeighborsClassifier(n_neighbors=7)),
        ("knn9", KNeighborsClassifier(n_neighbors=9))
    ]

    ensemble = VotingClassifier(
        estimators=models,
        voting="hard"
    )

    return ensemble


# Step 2 — Create the models for comparison

def create_hard_voting_models():
    models = {
        "KNN-1": KNeighborsClassifier(n_neighbors=1),
        "KNN-3": KNeighborsClassifier(n_neighbors=3),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "KNN-7": KNeighborsClassifier(n_neighbors=7),
        "KNN-9": KNeighborsClassifier(n_neighbors=9),
        "Hard Voting": create_hard_voting_model()
    }

    return models


# Step 3 — Evaluate the models

models = create_hard_voting_models()
results = []
names = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )

    results.append(scores)
    names.append(name)

    print(
        f"{name:12s} "
        f"Mean = {scores.mean():.3f} | "
        f"Std = {scores.std():.3f}"
    )


# Step 4 — Visualise

plt.figure(figsize=(10, 5))

plt.boxplot(
    results,
    tick_labels=names,
    showmeans=True
)

plt.ylabel("Accuracy")
plt.title("Hard Voting vs Individual KNN Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


#
#We want higher accuracy
#So the orange line is the median accuract the miidle accuracy so eg for knn1 50% result are above 0.88 and 50% below 0.88

#Green is the mean the average accuracy we want higher average accuracy
#Overall, the results show that increasing the value of K improves the accuracy of the KNN models.
"""We want higher accuracy
So the orange line is the median accuract the miidle accuracy so eg for knn1 50% result are above 0.88 and 50% below 0.88

Green is the mean the average accuracy we want higher average accuracy
#- Overall, the results show that increasing the value of K improves the accuracy of the KNN models.
-Hard Voting has the highest median accuracy at around 0.91 (91%).
KNN-7 and KNN-9 have similar median performance, around 90.5ex
s median accuracy is around 0.91, compared with about 0.88–0.905 for the individual models.
 It also has a relatively good and consistent spread of results. This suggests that combining the KNN models through hard voting improves predictive performance compared with using most individual models alone.


#complete this tabel
Approach	      Ensemble strategy	    How are models combined?	Sequential or parallel?
AdaBoost	       Boosting	            Models are trained one after another, with later models focusing more on previously misclassified examples	Sequential
Random Forest	   Bagging	            Multiple decision trees are trained on different random samples/features, then their predictions are combined	Parallel
KNN Bagging	       Bagging	            Multiple KNN models are trained on different bootstrap samples, then their predictions are combined	Parallel
Soft Voting	       Voting	            Models' predicted probabilities are averaged/weighted, and the class with the highest combined probability is selected	Parallel
Hard Voting	       Voting	             Models vote for a class, and the class receiving the most votes wins	Parallel

bagging=reduce variance
boosting- reduce bias by thinking about weaknesses
hardvoting-combine prediciton/highest vte win
soft cobine probabilitie shighest probability wins

f1 score:Model B has a better balance between precision and recall, so it is likely performing better across the classes
precision=true positive + recall how many positive you found = f1 score also

"""

