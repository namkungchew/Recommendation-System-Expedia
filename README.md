# Project 3: 
# Bitcoin - Ethereum Post Classifier

## Table of Contents:

1. Background
1. Problem Statement
1. Executive Summary
1. Data Cleaning
1. Models
    1. Preprocessing
    1. Different Combinations of Vectorizers and Classifiers
    1. Fine tuning and selecting final model
    1. Final Choosen Model: CountVectorizer with MultinomialNB
1. Final Model
1. Conclusion and Recommendations
1. References and Data Sources

## 1. Background
We have been engaged by a local trading exchange who is looking to issue a new cryptocurrency future contract. They are currently deciding between a Bitcoin future contract or a Ethereum future contract. They are interested to monitor the posts on their user discussion forum on cryptos and identify these posts as Bitcoin / Ethereum posts. They believe the crypto with the greater amount of post will be the contract that will have the larger interest and the future contract on it will be better received and have generate more liquidity faster.

---
## 2. Problem Statement
We are tasked with building a model to identify and classify the posts into Bitcoin or Ethereum posts. We decided to train our model using posts on Reddit on the subtopics Bitcoin and Ethereum.

---
## 3. Executive Summary
To achieve our goal of identifing posts on our stalkholder forum, we have to create a NLP classification model. To train our model, we decided to use posts on Reddit on the subtopics of Bitcoin and Ethereum. This is because the posts would be similar to those in our forum and that Reddit have readily available a much bigger amount of data and these are already classified. The metric that we will be most concerned about will be Accuracy.

Our strategy on building this NLP classification model is to try various combinations of stemming, lemmitzing, Vectorisations and Classifiers. We find that our results weren't much different between stemming and lemmitizing so we chose stemming to cut down the number of features and increase independence between features which is the assumption for Naive Bayes (and which is not true most of the time in NLP but used as it simplify the calculations). We tried CountVectorizer, TfidfVectorizer, Hashing Vectorizer. Hashing Vectorizer doesn't provide much benefit as our dataset is small and it is hard to analyse the important features. CountVectorizer and TfidfVectorizer works better with different classifiers. For the classifiers, we tried Logistic Regression, Multinomial Naive Bayes, K Nearest Classifier and Random Forest Classifier. K Nearest Classifier doesn't perform well. Logistic Regression and Random Forest accuracy are reasonable but Multinominal Naive Bayes is the best and it is the model whereby we can control the variance down without sacrificing the accuracy of the model much.

Our best model is a Multinominal Naive Bayes Model with Count Vectoriser. This model achieve an accuracy of 84.5% on unseen data and 88% on seen data. We feel that this model is the best because firstly it is the model with the highest accuracy. It is not overfitting and have reasonable variance. Our best params are the following: 'cvec__max_df': 0.1, 'cvec__max_features': 1000, 'cvec__min_df': 3, 'cvec__ngram_range': (1, 2), 'nb__alpha': 0.5. When we look the top words that differientate between Bitcoins and Ethereum. We can resonate with them (more of these in the final model section). The posts that were misclassified are mostly spam posts. Also it is the fastest to tune and compute over GridSearchCV when we have to refit our model with new data in the future.

In conclusion, we believe our model is adequate to solve the problem statement.

Going forward, as we are only training our model with 1309 data points with 4541 features, we have to cap the number of features to prevent overfitting. When we have more data, we can improve our accuracy by refitting it to our model and relaxing some of the hyperparameters used to control the overfitting.

---
## Data Dictionary
Data have been scapper from Reddit by 'Bitcoin - Reddit Scrapper.ipynb' and 'Ethereum - Reddit Scapper.ipynb' in the code folder and saved as csv files in the datasets folder.

---
## 6. Conclusion and Recommendations
|             | CountVectorizer with MultinomialNB | Tfidf Vectorizer with Logistic Regression | CountVectorizer with RandomForestClassifier |
|-------------|------------------------------------|-------------------------------------------|---------------------------------------------|
| Params      | 'cvec__max_df': 0.2                |  'tfidf__max_df': 0.2                     | 'cvec__max_features': 1000                  |
|             | cvec__max_features': 1000          | 'tfidf__max_features': 1000               |  'cvec__ngram_range': (1, 2)                |
|             | cvec__min_df': 2                   | tfidf__min_df': 3                         | rt__max_depth': 35                          |
|             | cvec__ngram_range': (1, 1)         | 'tfidf__ngram_range': (1, 2)              | 'rt__min_samples_leaf': 1                   |
|             | nb__alpha': 0.4                    | ''lr__C': 1,                              |                                             |
| Train Score | 0.881                              | 0.901                                     | 0.897                                       |
| Test Score  | 0.845                              | 0.823                                     | 0.795                                       |

Recap: We tried various models. Logistic Regression produce pretty consisent accuracy. Multinominal Naive Bayes worked the best. K Nearest Neighbours Classifier doesn't seem to perform well. Random Forest Classifier is around the same accuracy as Logistic Regression but still not better than Multinominal Naive Bayes. HashingVectorizer doesn't give much benefit and have more disadvantages for this dataset. We tried to fine tune 3 model: Model 2: MultinominalNB with CountVectorizer, Model 4: LogisticRegression with TfidfVectorizer and Model 10: Random Forest with CountVectorizer.

One of the challenges we face during the course of this project is that our models overfit most of the times and we have to find ways and trying different hyperparameters to prevent the model from overfitting without sacrificing accuracy. For our final 3 models, we discover that Multinominal respond best to our efforts to prevent overfitting. It was able to increase accuracy of the test score while minimizing the difference between our train and test score to around 4% and improving our cross val score. We find it hard to prevent overfitting on the Random Forest model as once we limit the max depth or increase the minimum sample leaf, our test score dropped drastically from 0.82 to 0.795. For the Logistic Regression model, we managed to reduce to 8% difference between the train and test score and still achieve 0.82 test score.

We choose Multinominal Naive Bayes with CountVectorizer as our final model as it is the model that still maintain its highest accuracy score and also is not overfitted. Also it is the fastest to tune and compute over GridSearchCV. Our best params are the following: 'cvec__max_df': 0.1, 'cvec__max_features': 1000, 'cvec__min_df': 3, 'cvec__ngram_range': (1, 2), 'nb__alpha': 0.5

We are satisfied with our results that it 84.5% accurate which beat the baseline accuracy by 24.8% and is better than the other models by 2-3%. Also the train score and test score are not significantly different thus not overfitting. This meet our goal of being able to differentiate and classify posts into Bitcoins posts and Ethereum posts reasonably well. We also resonate with the most important features in this model and they are similar with the most important features in other models as well. A look at the posts which are misclassified show these posts mainly outliers which are either spam or didn't belong to either classes which we cannot differentiate with our human brains either. In short, we believe our model to be adequte to solve the problem statement.

Going forward, since most of our models show overfitting due to the larger number of features to our dataset, we can improve our accuracy by gathering more data given time and refitting it to our model and relaxing some of the hyperparameters used to control the overfitting.

---
## 7. References and Data Sources
1. https://www.reddit.com/
