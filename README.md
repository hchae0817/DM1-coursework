# COVID Tweets Sentiment Analysis Project
This project involves the analysis of COVID-related tweets to classify their sentiment, extract insights, and build a machine learning model to predict the sentiment of future tweets. The project uses techniques from Natural Language Processing (NLP) and machine learning, specifically focusing on tokenization, stemming, stopword removal, and classification using a Multinomial Naive Bayes classifier.

## Project Description
The goal of this project is to analyze a corpus of coronavirus-related tweets and perform the following tasks:

##  Sentiment Analysis:
- Compute the possible sentiments that a tweet may have, determine the second most popular sentiment, and find the date with the greatest number of extremely positive tweets.
- Tokenization & Text Preprocessing: Tokenize the tweets into words, count the total number of words and distinct words, and identify the 10 most frequent words in the corpus. Apply various preprocessing steps to improve the quality of the text data.
## Text Cleanup: 
- Remove stop words, filter out words with fewer than or equal to 2 characters, and apply stemming to reduce words to their root form.
## Term-Document Matrix: 
- Store the tweets corpus in a NumPy array and produce a sparse representation of the term-document matrix using CountVectorizer.
## Multinomial Naive Bayes Classifier: 
- Build and train a Naive Bayes classifier on the preprocessed text data. Tune hyperparameters to optimize classification accuracy.

# Project Structure
The project is structured as follows:

- coronavirus_tweets.py: Contains the code to preprocess the tweets, perform tokenization, remove stopwords, and apply stemming.
- tweets_corpus.npy: A NumPy array storing the tweets corpus, which is used to build the term-document matrix.
- term_document_matrix_sparse.npy: A sparse representation of the term-document matrix.
- naive_bayes_classifier.py: Implements the Multinomial Naive Bayes classifier and evaluates its performance.


# Workflow
1. Sentiment Analysis
The sentiment of each tweet is computed, and the following tasks are completed:

Calculate the overall possible sentiments in the dataset.
Determine the second most popular sentiment.
Identify the date with the highest count of extremely positive tweets.
2. Tokenization and Text Analysis
We tokenize the tweets into individual words and perform basic text analysis:

Count the total number of words (including repetitions).
Count the number of distinct words in the corpus.
Identify the 10 most frequent words.
3. Text Cleanup
Before proceeding to build the term-document matrix, we clean the text data by:

Removing common stop words (e.g., 'the', 'is', 'and').
Filtering out words with 2 characters or fewer to improve the quality of the text data.
Stemming the words to reduce them to their root form (e.g., 'running' becomes 'run').
After this, we recompute the 10 most frequent words in the modified corpus and observe how the results change after cleaning the data.

4. Term-Document Matrix
Using CountVectorizer from scikit-learn, we create a sparse representation of the term-document matrix:

python
Copy code
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tweets)  # tweets is the cleaned list of tweet texts
This matrix stores the frequency of each word across all tweets.

5. Multinomial Naive Bayes Classifier
We train a Multinomial Naive Bayes classifier to predict the sentiment of each tweet. The model is trained on the preprocessed tweets and evaluated using classification accuracy.

To tune the model, we modify the parameters of CountVectorizer to control the range of frequencies and the number of words included in the term-document matrix. The goal is to achieve the highest classification accuracy possible.

6. Hyperparameter Tuning
Tune the parameters of CountVectorizer:

min_df: Minimum number of documents a word must appear in.
max_df: Maximum frequency a word can appear in.
max_features: Limit the number of features (words) to include.
Example:

python
Copy code
vectorizer = CountVectorizer(min_df=5, max_df=0.9, max_features=1000)
Experiment with different values of these parameters to optimize the model's accuracy.

7. Training Accuracy
Once the model is trained, its training accuracy is calculated. The accuracy reflects how well the model classifies tweets into different sentiments.

## Observations
After cleaning and preprocessing the tweets, the most frequent words in the corpus may change significantly. The removal of stop words, short words, and stemming can lead to more meaningful and distinctive words being identified.

The accuracy of the Naive Bayes classifier may vary depending on the preprocessing steps and the tuning of hyperparameters. It's crucial to experiment with different settings to achieve the best model performance.

## Conclusion
This project provides a thorough analysis of COVID-related tweets, including sentiment analysis, text preprocessing, and classification. By using CountVectorizer and a Multinomial Naive Bayes classifier, the project builds a predictive model capable of classifying tweet sentiment with high accuracy.

## Future Work
Experiment with other machine learning models (e.g., Logistic Regression, Support Vector Machine).
Integrate additional features such as tweet metadata (e.g., tweet length, hashtags, etc.).
Expand the dataset with more recent tweets for better model generalization.
Feel free to modify or expand on this based on your specific implementation details and outcomes. Let me know if you need any changes!
