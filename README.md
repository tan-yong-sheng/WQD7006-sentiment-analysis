# Comparative Analysis of Text Representation Techniques for Sentiment Classification of Airline Tweets

With social media booming, people are sharing their thoughts and opinions online more than ever. This creates a huge stream of text data that can be really useful for understanding how people feel about different topics. That's where sentiment analysis comes in — it’s all about using computers to figure out whether something written in text is positive, negative, or neutral. But before any machine learning model can do that, we first need to turn those words into numbers it can actually understand. This step, called text representation or feature extraction, plays a big role in how well the sentiment analysis works.

Using machine learning techniques, this project focuses on evaluating and comparing various text representation techniques using a dataset of airline-related tweets. The methods assessed range from traditional approaches such as Bag-of-Words (BoW) and TF-IDF, N-grams, to more advanced models like Word2Vec, and transformer-based embeddings such as Sentence-BERT (SBERT) and LLM embeddings. These features are used to train classification models including Logistic Regression, Decision Tree, and XGBoost. The study aims to identify which combinations of representation and model deliver the best performance, with additional tuning to further optimize results.


## Problem Statement
Airline reviews posted on social media platforms, particularly Twitter, are composed of unstructured, informal textual data. Tweets are often short and filled with slang, typos, and emojis, which makes it hard to automatically figure out if a customer is happy or upset with the airline companies.

Such short texts pose unique challenges for sentiment analysis, as they contain limited context and sparse linguistic cues. This makes it crucial to use effective text representation techniques that can capture meaningful information from minimal content.

To address this issue, the study aims to evaluate and compare various text representation methods—including traditional approaches like Bag-of-Words, N-grams, and TF-IDF, as well as advanced embeddings such as Word2Vec, SBERT, and large language model (LLM) embeddings—on short tweets. The goal is to identify which representations enable machine learning models to most accurately classify tweet sentiments into positive, negative, or neutral categories.

## Research Goals
The goals for this research are to:
- To evaluate and compare the performance of sentiment analysis classifier models across various text representation techniques, including Bag-of-Words, N-grams, TF-IDF, Word2Vec, SBERT, and large language model (LLM) embeddings.
- Evaluate the best combination of a text method and a machine learning model to get the most accurate results for airline tweets.

This research aims to provide a structured framework for extracting sentiment insights from unstructured airline feedback, ultimately supporting more responsive and data-driven decision-making in the airline industry.

## Our Notebooks

To prepare our textual data for sentiment classification, we first conducted an Exploratory Data Analysis (EDA). 
- [EDA notebook](./notebook/01_intro/eda.ipynb)

We've also explored various text representation techniques for our sentiment analysis classifier, which uses three sentiment labels: positive (2), neutral (1), and negative (0), as follows:

- [Bag of Words](./notebook/02_text-representation-comparison/01_bow.ipynb) : Feature extraction using Bag-of-Words with max_features = 10,000
- [Term Frequency-Inverse Document Frequency (TF-IDF)](./notebook/02_text-representation-comparison/02_tf_idf.ipynb) : Feature Extraction using TF-IDF with max_features = 10,000
- [Ngrams with bag of words](./notebook/02_text-representation-comparison/03_ngrams_bow.ipynb): Extracted unigram to trigram features using the Bag-of-Words model with a vocabulary limited to the top 10,000 features. This method captures raw frequency counts of each n-gram across documents.
- [Ngrams with TF-IDF](./notebook/02_text-representation-comparison/04_ngrams_tfidf.ipynb): Extracted unigram to trigram features using TF-IDF weighting, also capped at the top 10,000 features. This method accounts for the relative importance of terms across the entire corpus by down-weighting frequent but less informative n-grams.
- [Word2vec](./notebook/02_text-representation-comparison/05_word2vec.ipynb) : Using Word2Vec embedding model with dimension size of 300
- [SBERT embedding](./notebook/02_text-representation-comparison/06_sbert_embedding.ipynb): Using sbert embedding model called `all-MiniLM-L6-v2` with dimension size of 384
- [LLM embedding](./notebook/02_text-representation-comparison/07_llm_embedding.ipynb) : Using LLM embedding model called `text-embedding-004` from Google's Gemini with dimension size of 768

Also, we have also tuned our XGBoost model with LLM embedding for better performance. Check it out here: [Hyperparameter tuning notebook](./notebook/03_hyperparameter_tuning\llm_embedding_hyperparameter_tuning.ipynb)

## Our Dataset

This dataset describes the contents of the heart-disease diagnosis.

The dataset in this study is from [Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data), which is called Twitter US Airline Sentiment.

- Dataset: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data

- Variable Table:

| Original Dataset             | Data Type     | Description    |                                                         
|------------------------------|---------------|---------------------------------------------------------------------------------------------|
| tweet_id                     | ID            | A unique identifier for each tweet.                                                         | 
| airline_sentiment            | Categorical   | The sentiment expressed in the tweet (positive, neutral, negative).                         | 
| airline_sentiment_confidence | Numerical     | Confidence score in the sentiment label (0 to 1).                                           | 
| negativereason               | Categorical   | Reason for negative sentiment (e.g., "Late Flight", "Customer Service Issue").              | 
| negativereason_confidence    | Numerical     | Confidence score in the negative reason label (0 to 1).                                     | 
| airline                      | Categorical   | The airline mentioned in the tweet (e.g., United, Delta, etc.).                             | 
| airline_sentiment_gold       | Categorical   | Sentiment label by trusted annotator (gold standard).                                       | 
| name                         | Text          | Name of the user who posted the tweet.                                                      | 
| negativereason_gold          | Categorical   | Negative reason label by trusted annotator (gold standard).                                 | 
| retweet_count                | Numerical     | Number of times the tweet was retweeted.                                                    | 
| text                         | Text          | The full content of the tweet.                                                              | 
| tweet_coord                  | Geospatial    | Latitude and longitude coordinates where the tweet was posted, if available.                | 
| tweet_created                | Datetime      | Timestamp when the tweet was created.                                                       | 
| tweet_location               | Text          | Location specified in the user's profile.                                                   | 
| user_timezone                | Categorical   | Time zone specified in the user's profile.                                                  | 


## Variables Used for Modeling

| Feature                      | Data Type   | Description  |
|-----------------------------|-------------|--------------|
| **Target Variable: `encoded_sentiment`** | Categorical | This is an engineered variable derived from `airline_sentiment` for multi-class sentiment classification. It encodes sentiment as: 0 = Negative, 1 = Neutral, 2 = Positive. |
| **Feature: `text`**         | Text        | Contains consumer tweets about U.S. airlines. This field undergoes preprocessing, including removal of URLs and mentions (`@`), stopword removal, and stemming. We will be performing different feature extraction techniques such as bag of words, TF-IDF, word2vec, SBERT, and LLM embedding to compare the model performance under different text representation techniques. |
