📧 Spam Detection Using Machine Learning
📌 Overview

This project is a Spam Email/SMS Detection System that classifies messages into two categories:

Ham (0) – Legitimate messages

Spam (1) – Unwanted or fraudulent messages

We use Natural Language Processing (NLP) techniques for text preprocessing and multiple Machine Learning models to compare performance.

📂 Dataset

Source: spam.csv (SMS Spam Collection Dataset)

Shape: 5,572 rows × 5 columns (after cleaning → 5,169 rows × 2 columns)

Features:

v1 → Target label (ham or spam)

v2 → Message text

Additional extracted features:

num_chars → Number of characters

num_words → Number of words

num_sentences → Number of sentences

🛠️ Data Preprocessing

Drop unused columns: Unnamed: 2, Unnamed: 3, Unnamed: 4

Handle duplicates: Removed 403 duplicate rows

Label Encoding: Converted ham to 0 and spam to 1

Feature Engineering:

Character count

Word count

Sentence count

Text Transformation:

Lowercasing

Removing punctuation

Removing stopwords

Simple stemming

Keeping only alphanumeric tokens

Vectorization:

TF-IDF Vectorizer (max_features=3000)

📊 Exploratory Data Analysis (EDA)

Distribution of classes:

Ham: 87.3%

Spam: 12.6%

Spam messages are generally longer and contain more words than ham messages.

Strong positive correlation between num_chars and num_words.

🤖 Machine Learning Models Used
Model	Accuracy	Precision
Support Vector Classifier (SVC)	97.58%	97.47%
K-Nearest Neighbors (KNN)	90.32%	100%
Naive Bayes (MultinomialNB)	97.38%	100%
Decision Tree (DT)	92.16%	84.33%
Logistic Regression (LR)	95.45%	95.05%
Random Forest (RF)	97.09%	98.21%
AdaBoost (ABC)	90.13%	90.90%
Bagging Classifier (Bgc)	96.32%	90.32%
Extra Trees Classifier (ETC)	97.48%	97.45%
Gradient Boosting (GBDT)	94.29%	97.59%
XGBoost (xgb)	96.51%	94.73%
🏆 Best Performing Models

SVC (Best overall accuracy)

Naive Bayes (Perfect precision — no false positives)

Random Forest (Highest precision after Naive Bayes)

📌 How to Run

Clone the repository:

git clone https://github.com/nikitaphartiyal/spam-detection.git


Install dependencies:

pip install -r requirements.txt


Open the Jupyter Notebook or run in Google Colab:

jupyter notebook spam_detection.ipynb


Load dataset and execute all cells.

📦 Dependencies

Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

nltk

xgboost

📈 Future Improvements

Use advanced NLP techniques (Word2Vec, BERT)

Implement deep learning models (LSTM, Transformers)

Build a web-based interface for real-time spam detection
