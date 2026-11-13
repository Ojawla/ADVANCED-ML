# Standard imports and settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
#aded
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
#import nltk
nltk.download('stopwords')
nltk.download('punkt') # for tokenization if needed
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

'''orignialimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt') # for tokenization if needed
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))'''









imdb = pd.read_csv('IMDB Dataset.csv') # Colab path; adjust if needed
print("Shape:", imdb.shape)
print(imdb['sentiment'].value_counts())
imdb.head(6)


#sentiment is if review is positive or negative
#display 3 tandom reviewd

for i in imdb.sample(3).index:
    print("=== SAMPLE ===")
    print(imdb.loc[i,'review'][:500], "\n---label:", imdb.loc[i,'sentiment'])
'''I loaded the IMDB dataset using Pandas and examined its dimensions to understand the dataset size and structure. I used value_counts() to examine the distribution of positive and negative sentiment labels and identify whether the dataset was balanced. I also inspected the first six records and displayed three randomly selected reviews to understand the raw text data and corresponding sentiment labels.'''

#cleaning text
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_text(text, do_lemmatize=True, remove_digits=True):
#Remove HTML Tags
    text = BeautifulSoup(text, "html.parser").get_text()
# 2. remove [..] annotations IN SQUARE BRACKETS
    text = re.sub(r'\[[^\]]*\]', ' ', text)
# 3. keep letters (convert digits optionally)
    if remove_digits:
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
# 4. lowercase and tokenize SPLIT INTO WORDS
    tokens = text.lower().split()
# 5. remove stopwords BASICALLY BRING EACH WORD BACK TO ITS ROOTS
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Apply to a small sample first i wated to verify if all cleaning fucntions wer applied

imdb_sample = imdb.sample(2000, random_state=42).copy()
imdb_sample['clean'] = imdb_sample['review'].apply(clean_text)
print(imdb_sample[['review','clean']].head(3))


#: Apply cleaning to whole dataset

imdb['clean'] = imdb['review'].apply(clean_text)
imdb.to_csv('output.csv', index=False )

#check for missing values
print(imdb['clean'].head())
print('total missing')
print(imdb['clean'].isna().sum())

#if there were na fill with empty string
imdb['clean'] = imdb['clean'].fillna('')



#Feature extraction: CountVectorizer & TF-IDF
#convert your cleaned IMDB movie reviews into numbers that a machine learning model can proces
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
MAX_FEAT = 5000 # vectorisers to select a maximum of 5,000 features (words or phrases) from your dataset.
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) #1,2ngram_range=(1,2) means the vectoriser considers both individual words (unigrams) and pairs of consecutive words (bigrams) to capture more context in the text.
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2))
X_bow = bow_vec.fit_transform(imdb['clean'])
X_tfidf = tfidf_vec.fit_transform(imdb['clean'])
print("BOW shape:", X_bow.shape)
print("TFIDF shape:", X_tfidf.shape)



#its output BOW shape: (50000, 5000) TFIDF shape: (50000, 5000) both have 50k reviews an d5k columns
#bow and tdf are both techniques used to convert text to number for ml to understand
#bow focuses on frequency eg amaing appears 3 times value=3
#how important word is in a review more distijnctive words receive higher weights
"""
#this will allow you inspect top features
print(bow_vec.get_feature_names_out()[:30])
"""
#Train test split
#here we just divided the data
from sklearn.model_selection import train_test_split
#label encoding
y = imdb['sentiment'].map({'positive':1, 'negative':0}).values #het sentiment column and convert positive 1 negtive 0

# not here but if it was 3 class(y = imdb['sentiment'].map({'negative': 0,'neutral': 1,'positive': 2}).values

#X_tr_bow:80% of the BoW features used to train the model
#X_tr_bow:80% of the BoW features used to train the model,X_te_bow:20% of the BoW features used to test the mode
# y for training and y for testing
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)

#secon technique to convert the dta
X_tr_tfidf, X_te_tfidf, y_tr_tfidf, y_te_tfidf = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)
# train test "The dataset contains 50,000 reviews represented by 5,000 features. I split the data into 80% training data (40,000 reviews) and 20% testing data (10,000 reviews). The training data is used to teach the model, while the testing data evaluates how well it performs on unseen reviews."



#stratify keeps teh split of positive and negative reviews
#we have 2 different representation of same dataset bow and tdf idf
#we spli both into training and testing split

#baseline model
#why Multinomial Naive Bayes (MNB)
#I selected Multinomial Naive Bayes because it is a commonly used classification algorithm for text-based data. It works well with word-frequency features such as Bag of Words and TF-IDF and uses probabilities to classify reviews into positive or negative sentiment classes. It also provides a computationally efficient baseline for comparison.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#The model learns from the training data, but we judge how well it performs using the testing data.


#nlp is a field of a not a specific midel
#so you use the models we have learnt before still eg logistic regressiona nd naive bayes for classification  random forest too for classiciation
# MultinomialNB on BOW
mnb = MultinomialNB()
mnb.fit(X_tr_bow, y_tr)
mnb_pred = mnb.predict(X_te_bow)
print("MNB (BOW) acc:", accuracy_score(y_te, mnb_pred))
print(classification_report(y_te, mnb_pred))


#Both BOW and TF-IDF are text vectorisation techniques. They convert text (words) into numerical values so machine learning models can process it
# Logistic Regression on BOW
lr = LogisticRegression(max_iter=1000)
lr.fit(X_tr_bow, y_tr)
lr_pred = lr.predict(X_te_bow)
print("LR (BOW) acc:", accuracy_score(y_te, lr_pred))
print(classification_report(y_te, lr_pred))

#so w run both model we train both models and run test data
#acccuracy: naivebayes- accuracy:84.07 87
#precision:0.84  0.88 #recall 0.84 0.88  f1score:0.84 and 0.88
#When using Bag of Words (BOW) vectorisation, Logistic Regression outperformed Multinomial Naive Bayes across all evaluation metric
#Both models performed equally across their respective precision, recall and F1-score metrics for the two sentiment classes, suggesting balanced classification performance.
#Precision measures how many of the reviews the model predicted as positive were actually positive
#Recall measures how many of the reviews that were actually positive the model successfully identified as positive

#repeat both models on tf-idf features
# Multinomial Naive Bayes on TF-IDF
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tr_tfidf, y_tr)
mnb_tfidf_pred = mnb_tfidf.predict(X_te_tfidf)

print("MNB (TF-IDF) acc:", accuracy_score(y_te, mnb_tfidf_pred))
print(classification_report(y_te, mnb_tfidf_pred))

# why logistic regression
#'''I selected Logistic Regression because it is suitable for binary classification problems, such as predicting whether a movie review is positive or negative. It performs effectively with high-dimensional text data represented using BOW and TF-IDF.'''
# Logistic Regression on TF-IDF
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_tr_tfidf, y_tr)
lr_tfidf_pred = lr_tfidf.predict(X_te_tfidf)
print("LR (TF-IDF) acc:", accuracy_score(y_te, lr_tfidf_pred))
print(classification_report(y_te, lr_tfidf_pred))

# triane dwith training data then tested with testing adata
##Logistic Regression achieved better classification performance than Multinomial Naive Bayes across both text representations. TF-IDF also outperformed Bag of Words for both algorithms
#The best-performing combination was Logistic Regression with TF-IDF, achieving 89.64% accuracy and an F1-score of 0.90.
#This suggests that weighting words according to their importance provided a more effective representation for sentiment classification in this experiment
#I set max_iter=1000 to allow the Logistic Regression algorithm sufficient iterations to converge during training.

#step 7 trying to improve models acuracy one of the models
#best was lr using tf-idf try to improve accuracy further 89.64

#change vectoriser td idf setting
# 1. Create a new vectoriser changed We allow the computer to consider up to 8000 words instead of 5k include single word pair of word an group of 3
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1,3))

# STEP 7 - ROUND 1: CHANGE TF-IDF VECTORIZER

# 1. Create a new TF-IDF vectorizer
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1,3))

# 2. Convert cleaned reviews into numerical features
X2 = vec2.fit_transform(imdb['clean'])

print("New TF-IDF shape:", X2.shape)


# 3. Split data into training and testing sets
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X2, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_tr2.shape)
print("Test shape:", X_te2.shape)

# 5. Train Logistic Regression
lr2 = LogisticRegression(max_iter=1000)
lr2.fit(X_tr2, y_tr2)

lr2_pred = lr2.predict(X_te2)

print("\nLR (TF-IDF 8000, 1-3) accuracy:",
      accuracy_score(y_te2, lr2_pred))

print(classification_report(y_te2, lr2_pred))

#confusion matrix lr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_lr = confusion_matrix(y_te, lr_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=['Negative', 'Positive']
)

disp.plot(cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()




#way2 try different model
#try different model
#sgd classifier
'''
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(max_iter=500, tol=1e-3, random_state=42)
sgd.fit(X_tr_tfidf, y_tr)

sgd_pred = sgd.predict(X_te_tfidf)

print("SGD Accuracy:", accuracy_score(y_te, sgd_pred))


#Each group submit best accuracy and short note: what change gave the improvement and hypothesized why
#check if accuracy increases
'''
#adaboost
#why adaboost
'AdaBoost is an ensemble learning technique that combines multiple weak learners to create a stronger predictive model. Each new learner focuses more on the observations that previous learners classified incorrectly.'
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
#label encoding
y = imdb['sentiment'].map({'positive':1, 'negative':0}).values #het sentiment column and convert positive 1 negtive 0
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
X_tr_tfidf, X_te_tfidf, y_tr_tfidf, y_te_tfidf = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)

#training model
max_depth=3 # t just allows each decision tree to learn more complex patterns.
boosting_model = AdaBoostClassifier(
estimator=DecisionTreeClassifier(
max_depth=3,
random_state=1
),
n_estimators=200,#o create 200 decision trees within the Random Forest. Combining predictions from multiple trees helps reduce the reliance on an individual tree and can improve the stability of the model's predictions.
random_state=1
)
boosting_model.fit(X_tr_tfidf, y_tr_tfidf)
boost_predictions = boosting_model.predict(X_te_tfidf)
boost_accuracy = accuracy_score(
y_te_tfidf,
boost_predictions
)
#calculate f1 score
boost_f1 = f1_score(
y_te_tfidf,
boost_predictions
)
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_te_tfidf, boost_predictions)
print(classification_report(y_te_tfidf, boost_predictions))
print(cm)
#ti visualis econfusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_te_tfidf, boost_predictions)

# Visualise confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)

plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.title('AdaBoost Confusion Matrix')
plt.show()

#comparing  both models
#adaboost and lr
from sklearn.metrics import accuracy_score

lr_accuracy = accuracy_score(y_te_tfidf, lr_tfidf_pred)
ada_accuracy = accuracy_score(y_te_tfidf, boost_predictions)

print("Logistic Regression:", lr_accuracy)
print("AdaBoost:", ada_accuracy)

#create bar chart
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'AdaBoost']
accuracies = [lr_accuracy, ada_accuracy]

plt.figure(figsize=(8, 5))

plt.bar(models, accuracies)

plt.title('Model Accuracy Comparison')
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

plt.show()

#confusion matrix

#checked most important feature for lr
# Most informative features for LR (if using scikit-learn LR)
def show_top_coef(classifier, vectorizer, n=20):
# assumes binary LR with coef_ shape (1, n_features)
    feat_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]
    print("Top positive features:\n", feat_names[top_pos])
    print("Top negative features:\n", feat_names[top_neg])
show_top_coef(lr, bow_vec, n=20)

'explain gistic Regression identified words such as excellent, amazing and brilliant as strong indicators of positive sentiment, while words such as worst, boring and terrible were associated with negative sentiment. This suggests the model learned meaningful relationships between vocabulary and sentiment'





#Word clouds: Show the most frequently used words in positive and negative reviews. For example, positive reviews might contain amazing,

from wordcloud import WordCloud
import matplotlib.pyplot as plt
#Feature importance (Logistic Regression): Shows which words have the strongest influence on the model's positive or negative predictions.
pos_text = ' '.join(imdb[imdb.sentiment=='positive']['clean'].tolist())
neg_text = ' '.join(imdb[imdb.sentiment=='negative']['clean'].tolist())
wc_pos = WordCloud(max_words=200).generate(pos_text)
wc_neg = WordCloud(max_words=200).generate(neg_text)
plt.figure(figsize=(10,4)); plt.imshow(wc_pos); plt.axis('off'); plt.title('Positive words')
plt.figure(figsize=(10,4)); plt.imshow(wc_neg); plt.axis('off'); plt.title('Negative words')


#COFISUION MATRIX
#1ST BOXX TRUE NEGATIVE AND BEXT FALSE POSITIVE
#SECOD ROW FITST BOX IS FALL NEGATIBE AND 4TH BOX IS TRUE POSITIVE
#The confusion matrix indicates that Logistic Regression correctly classified 4,387 negative observations and 4,391 positive observations. However, it incorrectly classified 613 negative observations as positive and 609 positive observations as negative. The model achieved an overall accuracy of 87.78%, suggesting relatively good classification performance

#"Comparing both models, AdaBoost outperformed Logistic Regression across all four evaluation metrics.
# AdaBoost achieved higher accuracy, suggesting better overall classification performance. Its higher recall also indicates that it identified a greater proportion of actual positive observations. Furthermore, its higher F1-score demonstrates a better balance between precision and recall. Therefore, AdaBoost appears to be the stronger model for this classification task."
#RECALL WA SIMPORTANT BECAUSERecall can be particularly important because you want to catch fraudulent transactions.
#PRECISON WAS IMPORTANT BEACUSE Recall can be particularly important because you want to catch fraudulent transactions.
'''
#123
#STRUCTURE
Baseline model evaluation
Explain confusion matrix → Accuracy → Precision → Recall → F1-score.

# REMEMBER Fraud detection: Recall can be particularly important because you want to catch fraudulent transactions
#Spam detection: Precision may matter because you don't want legitimate emails classified as spam.
Advanced model evaluation

Explain confusion matrix → Accuracy → Precision → Recall → F1-score.



Model comparison

Compare the four metrics and identify which model performs better.



Final recommendation

Select the stronger model, explaining why based on the problem's priorities.

'''



























'''
Page
Page number
3
of 6
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 1/6
Practical 2(additional)
Advanced Machine Learning Technologies for AI
Level 6 – Data Science & AI
Topic tutorial: NLP(Natural Language Processing)
Dr Gopal Jamnal
Introduction:
This step-by-step guide is designed for a 2-hour practical session on Natural Language Processing (NLP) using
the IMDB Movie Reviews dataset. You will follow structured Python tasks to explore text preprocessing,
feature extraction, and sentiment classification, with checkpoints, coding exercises, and reflection questions
to deepen understanding.
Prereqs (run once at start):
# Standard imports and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt') # for tokenization if needed
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
Step 1 — Load & quick exploration (10 mins)
Objective: Confirm shape, class balance, and peek at raw reviews so students understand the data.
imdb = pd.read_csv('/content/IMDB Dataset.csv') # Colab path; adjust if needed
print("Shape:", imdb.shape)
print(imdb['sentiment'].value_counts())
imdb.head(6)
Checkpoint:
• Expect 50,000 rows and 2 columns review and sentiment.
• There should be roughly 25k positive and 25k negative.
Mini task: Display 3 random reviews:
for i in imdb.sample(3).index:
print("=== SAMPLE ===")
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 2/6
print(imdb.loc[i,'review'][:500], "\n---label:", imdb.loc[i,'sentiment'])
Question : what is the sentiment of the 3 samples ?
Step 2 — Clean text (20 mins)
Objective: Implement a robust clean_text() function (HTML removal, lowercasing, non-alpha removal, optional
stemming/lemmatization, stopword removal).
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def clean_text(text, do_lemmatize=True, remove_digits=True):
# 1. HTML
text = BeautifulSoup(text, "html.parser").get_text()
# 2. remove [..] annotations
text = re.sub(r'\[[^\]]*\]', ' ', text)
# 3. keep letters (convert digits optionally)
if remove_digits:
text = re.sub(r'[^a-zA-Z\s]', ' ', text)
else:
text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
# 4. lowercase and tokenize
tokens = text.lower().split()
# 5. remove stopwords
tokens = [t for t in tokens if t not in stop_words]
# 6. lemmatize (optional)
if do_lemmatize:
tokens = [lemmatizer.lemmatize(t) for t in tokens]
return ' '.join(tokens)
# Apply to a small sample first (fast)
imdb_sample = imdb.sample(2000, random_state=42).copy()
imdb_sample['clean'] = imdb_sample['review'].apply(clean_text)
imdb_sample[['review','clean']].head(3)
Checkpoint:
• Compare raw vs cleaned text — many punctuation, HTML tags removed.
• Confirm clean is lowercase and stopwords are removed.
Challenge: Time-boxed: each pair tries (a) stemming with PorterStemmer, (b) lemmatization — record which yields shorter
results and which improves model later.
Troubleshooting: If NLTK errors, re-run downloads: nltk.download('stopwords'), nltk.download('punkt'),
nltk.download('wordnet').
Step 3 — Full dataset preprocessing
Objective: Apply cleaning to whole dataset (or to a subset if runtime is a concern). Use .sample() if time-limited.
imdb['clean'] = imdb['review'].apply(clean_text)
imdb.to_csv('/content/imdb_clean_short_debug.csv', index=False) # optional checkpoint save
Checkpoint: imdb['clean'].head() — no NaNs. If there are NaNs, fill with empty string: imdb['clean'] = imdb['clean'].fillna('').
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 3/6
Step 4 — Feature extraction: CountVectorizer & TF-IDF (15 mins)
Objective: Convert text → numerical features. Use max_features to keep runtime friendly.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
MAX_FEAT = 5000 # manageable for a class demo
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1,2))
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2))
X_bow = bow_vec.fit_transform(imdb['clean'])
X_tfidf = tfidf_vec.fit_transform(imdb['clean'])
print("BOW shape:", X_bow.shape)
print("TFIDF shape:", X_tfidf.shape)
Checkpoint:
• Students should see (50000, 5000) shapes (if whole dataset). If using sample, shape matches sample size.
Mini task: Inspect top features:
print(bow_vec.get_feature_names_out()[:30])
Challenge: Try ngram_range=(1,1) and (1,3) and compare feature counts.
Step 5 — Train / test split (5 mins)
Objective: Create reproducible train/test partition (80/20).
from sklearn.model_selection import train_test_split
y = imdb['sentiment'].map({'positive':1, 'negative':0}).values
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)
Checkpoint: Class proportions should be preserved in train/test.
Step 6 — Baseline models: Naive Bayes & Logistic Regression (20 mins)
Objective: Fit baseline classifiers and compare results.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# MultinomialNB on BOW
mnb = MultinomialNB()
mnb.fit(X_tr_bow, y_tr)
mnb_pred = mnb.predict(X_te_bow)
print("MNB (BOW) acc:", accuracy_score(y_te, mnb_pred))
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 4/6
print(classification_report(y_te, mnb_pred))
# Logistic Regression on BOW
lr = LogisticRegression(max_iter=400)
lr.fit(X_tr_bow, y_tr)
lr_pred = lr.predict(X_te_bow)
print("LR (BOW) acc:", accuracy_score(y_te, lr_pred))
print(classification_report(y_te, lr_pred))
Expected (typical) observation: MultinomialNB often yields ~0.75 accuracy and is a very strong baseline for sentiment; LR
may be slightly lower or similar depending on preprocessing and features.
Task: Record both accuracies and which model gives higher recall for positive class.
Challenge: Repeat both models on TF-IDF features and compare.
Step 7 — Small model competition
Objective: Lets make it gamified — groups try small tuning to beat baseline.
(Competition format):
• Groups of 3–4 get identical data.
• 3 rounds (10 minutes each): each group tries one change and reports accuracy:
1. Change vectorizer max_features or ngram range.
2. Add lemmatization vs stemming vs none.
3. Try a simple model change (SGDClassifier / PassiveAggressive / SVM) or adjust hyperparameters.
Examples to try (code snippets):
# change ngram or features
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1,3))
X2 = vec2.fit_transform(imdb['clean'])
# then split and run MNB/LR as above
# try SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(max_iter=500, tol=1e-3)
sgd.fit(X_tr_tfidf, y_tr)
print("SGD (tfidf) acc:", accuracy_score(y_te, sgd.predict(X_te_tfidf)))
Checkpoint: Each group submit best accuracy and short note: what change gave the improvement and hypothesized why.
Step 8 — Error analysis (10 mins)
Objective: Learn from mistakes — inspect misclassified reviews.
# Use MNB predictions as example:
mis = (y_te != mnb_pred)
mis_ix = np.where(mis)[0][:10] # first 10 errors
# map indices back to original rows after train_test_split:
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 5/6
# We need the test indices. Simpler: rebuild split with return indices:
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for tr_idx, te_idx in sss.split(imdb['clean'], y):
break
# Example: show misclassified examples
for i in mis_ix:
orig_i = te_idx[i]
print("ORIG INDEX:", orig_i)
print("Label:", imdb.loc[orig_i, 'sentiment'])
print("Pred:", "positive" if mnb_pred[i]==1 else "negative")
print(imdb.loc[orig_i, 'clean'][:400])
print("----")
Task: For 3 misclassified samples, say in 1 sentence why model likely failed (sarcasm, negation, rare words, long review with
mixed sentiment, etc.).
Step 9 — Visualization
Objective: Word clouds and most influential features (for LR).
from wordcloud import WordCloud
pos_text = ' '.join(imdb[imdb.sentiment=='positive']['clean'].tolist())
neg_text = ' '.join(imdb[imdb.sentiment=='negative']['clean'].tolist())
wc_pos = WordCloud(max_words=200).generate(pos_text)
wc_neg = WordCloud(max_words=200).generate(neg_text)
plt.figure(figsize=(10,4)); plt.imshow(wc_pos); plt.axis('off'); plt.title('Positive words')
plt.figure(figsize=(10,4)); plt.imshow(wc_neg); plt.axis('off'); plt.title('Negative words')
# Most informative features for LR (if using scikit-learn LR)
def show_top_coef(classifier, vectorizer, n=20):
# assumes binary LR with coef_ shape (1, n_features)
feat_names = vectorizer.get_feature_names_out()
coefs = classifier.coef_[0]
top_pos = np.argsort(coefs)[-n:]
top_neg = np.argsort(coefs)[:n]
print("Top positive features:\n", feat_names[top_pos])
print("Top negative features:\n", feat_names[top_neg])
show_top_coef(lr, bow_vec, n=20)
Task: Which features indicate positive sentiment? Which negative? Any surprising words?
Natural Language Processing – IMDB Movie
https://colab.research.google.com/drive/1boSSDwgR7NyUg-YmErMpaKBOxxNkXijx#scrollTo=CvcOVszb8XSq&printMode=true 6/6
Step 10 — Reflection with extension ideas
Group reflection prompts:
• Which preprocessing helped most and why?
• BOW vs TF-IDF — which performed better on your runs? Hypothesize why.
• Which errors are most common? How could we handle negation, sarcasm, or long reviews?
Extension tasks (for homework – if you can ):
• Implement bigram features only and test.
• Try a simple deep learning baseline: an embedding + 1-layer LSTM (use tf.keras in Colab). Note: show only for groups
that finish early.
• Use sklearn.pipeline.Pipeline to chain preprocessing + vectorizer + classifier and run a GridSearchCV for
hyperparameters (e.g., ngram_range, C for LR).
Summary questions:
1. Write one line: What was the most useful thing you learned?
2. Suggest one small change to improve the lab.
Appendix: FAQS
Common troubleshooting factors:
• Slow vectorizer / memory errors: reduce max_features to 2,000 or sample the dataset (imdb =
imdb.sample(20000)).
• ValueError: empty vocabulary from vectorizer — means clean column is empty or all removed by cleaning;
check imdb['clean'].str.len().describe() and handle empty strings.
• NLTK downloads fail: re-run nltk.download() or use !pip install nltk then restart runtime.
• Confusion matrix indexing issues: ensure you map train/test indices properly if you want to display original
text for misclassifications (use StratifiedShuffleSplit to preserve indices).'''