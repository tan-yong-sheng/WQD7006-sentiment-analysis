# Comparative Analysis of Text Representation Techniques for Sentiment Classification of Airline Tweets

With social media booming, people are sharing their thoughts and opinions online more than ever. This creates a huge stream of text data that can be really useful for understanding how people feel about different topics. That’s where sentiment analysis comes in — it’s all about using computers to figure out whether something written in text is positive, negative, or neutral. But before any machine learning model can do that, we first need to turn those words into numbers it can actually understand. This step, called text representation or feature extraction, plays a big role in how well the sentiment analysis works.

This project focuses on evaluating and comparing various text representation techniques using a dataset of airline-related tweets. The methods assessed range from traditional approaches such as Bag-of-Words (BoW) and TF-IDF, N-grams, to more advanced models like Word2Vec, and transformer-based embeddings such as Sentence-BERT (SBERT). These features are used to train classification models including Logistic Regression, Decision Tree, and XGBoost. The study aims to identify which combinations of representation and model deliver the best performance, with additional tuning to further optimize results.


## Problem Statement
- To compare model performance of sentiment analysis classifier model under different text representation techniques
- To 

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

<br/>

## Variables Used for Modeling

| Feature                      | Data Type   | Description  |
|-----------------------------|-------------|--------------|
| **Target Variable: `encoded_sentiment`** | Categorical | This is an engineered variable derived from `airline_sentiment` for multi-class sentiment classification. It encodes sentiment as: 0 = Negative, 1 = Neutral, 2 = Positive. |
| **Feature: `text`**         | Text        | Contains consumer tweets about U.S. airlines. This field undergoes preprocessing, including removal of URLs and mentions (`@`), stopword removal, and stemming. We will be performing different feature extraction techniques such as bag of words, TF-IDF, word2vec, SBERT, and LLM embedding to compare the model performance under different text representation techniques. |

<br/>
