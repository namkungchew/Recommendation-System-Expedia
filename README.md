Dataset and Models stored: https://drive.google.com/drive/folders/11S_7zzviD7ZyiiHQKZIimW5gIiGpf_yz?usp=sharing

# Project 5: 
# Expedia Hotel Recommendations
## Which hotel type will an Expedia customer book?

## Table of Contents:

### Part 1
1. Background
1. Problem Statement
1. Executive Summary
1. Exploratory Data Analysis and Data Cleaning
    1. Data Dictionary
    1. Preliminary Data Cleaning
    1. Preliminary Exploratory Data Analysis
    1. Feature Engineering
    1. More Exploratory Data Analysis
    1. Final Data Cleaning and processing the whole dataset
    
### Part 2
5. Final Dataset Preparation for Modelling
1. Modeling
    1. Preprocessing Functions
    1. Base Case
    1. Categorical Naive Bayes
    1. Random Forrest Classifier Model
    1. K Nearest Neighbor
1. TensorFlow Recommedation System
1. Conclusion and Recommendations
1. References and Data Sources

### Appendix
1. TensorFlow Recommedation System

## 1. Background
Expedia is a online travel hotel booking website. At this moment, they are using search parameters to adjust their hotel recommendations for each users. They would like to personalise their recommendations more for each user by employing AI.

---
## 2. Problem Statement
Using past data of each user, we are tasked to build a model that predicts the likelihood a user will stay at 100 different hotel groups i.e. a problem of classifying 100 different classes. Using this model, we hope to predict top 5 hotel groups that appeals more and is more likely to be booked by the user.

The metric for evaluating our model is Mean Average Precision @ 5 which is the area under the Precision and Recall curve for 5 predictions.

![image](https://user-images.githubusercontent.com/83707934/127964844-b4207c66-2b91-40ee-8275-ded33a427cab.png)


---
## 3. Executive Summary
Our objective in this project is to achieve a high MAP@5 score by prediction. The challenges that we experienced are firstly, we have to predict 100 different classes, this means that the probability of getting the correct prediction is small. Secondly, our features are all nominal categorical features. Some of the features have high cardinality. High Cardinality is a double edge sword for us, on one hand it provide variances to distinguish between the many hotel_cluster classes. We appreciate the variance especially in this case where we needed enough variance for distingushing between 100 different clasess. On the other, it messes around with the models whereby the model stalls when it is faced with new category that it have not seen before. We attempt to circumvent this issue by creating a new category, others, during training phrase to map all these unseen categories.

Although the features were represented by integars, as they were all nominal features, we cannot assume any linear relationship between the categories and the numbers representing them. We have to make sure our model is robust and give the same results even when we switch around the numbers representing the different categories. As there is no linear relationship, we did our EDA using Cramer's V with biased correction which gave us more meaningful results and return to us the association between the different features.

On the modeling, we tested various different models first using a small dataset which we filter out 10 classes and 100,000 data points from the main dataset. As expected, models which depends on linearity within the features are not doing well like, K-Nearest Neighbor or Logistic Regression. Due to high cardinality, Random Forest Classifier also didn't work too well as we had to do OneHotEncoding to the features to avoid RandomForestClassifier from treating them like continuous features. Also it is computationally expensive. The ML model that perform well is Categorical Naive Bayes which is still computationally inexpensive comparatively and it also managed our features relatively well. CategoricalNB obtained a MAP@5 score of 0.6036 on this smaller dataset. We also decided to try out the newest TensorFlow Recommendation System which was released in 2020 and use it for our prediction as doing a recommendation system was our main motivation in choosing this Kaggle Challenge. Our TFRS model achieve a MAP@5 score of 0.5695 on the testset.

We tried out the CategoricalNB model and the TensorFlow Recommendation System on the full dataset and achieve the Kaggle score of 0.2941 using the CategoricalNB model and 0.2194 on the TFRS model. Although the CategoricalNB model perform better than our TFRS model, it is because our CategoricalNB model have been tuned more finely than our TFRS model which still show signs of overfitting. Given more time, we could tune the TFRS model more finely and compare the results again.

That saying, we like the potential of our TFRS model. Despite us not yet having prevented it from overfitting, it already achieve a score close to our best model, the CategoricalNB model. Also there is ways that we can improve the model compared to the CategoricalNB Model.

We also like that the model can handle new categories with ease which is a challenge for all our other ML models including the CategoricalNB where we had to write a preprocessing step to categorical all new categories as 'others. Furthermore we can train this model by batches which will sort the issue of out of memory when training with huge datasets. This also means that we do not have to retrain the model when we have need data and we can keep improving the model when we have new data. In addition, the model is one of the models trained the fastest.

This model also have the flexibility to be reconfigured for different requirement. For example as mentioned earlier, our model is configured to recommend hotel cluster based on the user past preferences and searched destination. We can also reconfigure the model to recommend destinations and hotel cluster based on user past preferences by putting destination under the Candidate Model and using the destination dataset to find similarities between destinations.

In conclusion, we are satifised with the performance of our top 2 models especially the TFRS model which achieved a kaggle score of 0.2194 and beat the baseline score of 0.07. At the same time, the models can be further improved to achieve better scores which match the top scores in Kaggle.

Going forward, for the TFRS model, we can finetune the model to address overfitting using early stopping or regularization and also build a multitask TFRS model which include a ranking task to improve the prediction. For the CategoricalNB model, we can try balancing the different classes more by using an undersampling for some of the classes and undersampling for some of the classes within memory allowance and we can also explore using partial_fit in CategoricalNB to manage the memory usage. This will require the use of a custom pipeline.

---
## Data Dictionary

| Column name               | Description                                                                                                               | Data type |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------|
| date_time                 | Timestamp                                                                                                                 | string    |
| site_name                 | ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, Expedia.co.jp, ...)                                     | int       |
| posa_continent            | ID of continent associated with site_name                                                                                 | int       |
| user_location_country     | The ID of the country the customer is located                                                                             | int       |
| user_location_region      | The ID of the region the customer is located                                                                              | int       |
| user_location_city        | The ID of the city the customer is located                                                                                | int       |
| orig_destination_distance | Physical distance between a hotel and a customer at the time of search. A null means the distance could not be calculated | double    |
| user_id                   | ID of user                                                                                                                | int       |
| is_mobile                 | 1 when a user connected from a mobile device, 0 otherwise                                                                 | tinyint   |
| is_package                | 1 if the click/booking was generated as a part of a package (i.e. combined with a flight), 0 otherwise                    | int       |
| channel                   | ID of a marketing channel                                                                                                 | int       |
| srch_ci                   | Checkin date                                                                                                              | string    |
| srch_co                   | Checkout date                                                                                                             | string    |
| srch_adults_cnt           | The number of adults specified in the hotel room                                                                          | int       |
| srch_children_cnt         | The number of (extra occupancy) children specified in the hotel room                                                      | int       |
| srch_rm_cnt               | The number of hotel rooms specified in the search                                                                         | int       |
| srch_destination_id       | ID of the destination where the hotel search was performed                                                                | int       |
| srch_destination_type_id  | Type of destination                                                                                                       | int       |
| hotel_continent           | Hotel continent                                                                                                           | int       |
| hotel_country             | Hotel country                                                                                                             | int       |
| hotel_market              | Hotel market                                                                                                              | int       |
| is_booking                | 1 if a booking, 0 if a click                                                                                              | tinyint   |
| cnt                       | Numer of similar events in the context of the same user session                                                           | bigint    |
| hotel_cluster             | ID of a hotel cluster                                                                                                     | int       |

https://www.kaggle.com/c/expedia-hotel-recommendations/data

---
## 6. Conclusion and Recommendations
|                     | Small Dataset (10 classes, 100,000 data) |                               |                            |                          |                      |               |
|---------------------|------------------------------------------|-------------------------------|----------------------------|--------------------------|----------------------|---------------|
|                     | Baseline                                 | CategoricalNB (without SMOTE) | CategoricalNB (with SMOTE) | Random Forest Classifier | K Nearest Neighbours | Advanced TFRS |
| Trainset MAPK Score | 0.0725                                   | 0.7354                        | 0.7224                     | 0.3907                   | 0.5554               | 0.9676        |
| Holdset MAPK Score  | 0.0725                                   | 0.6036                        | 0.5913                     | 0.3789                   | 0.3213               | 0.5695        |

We tested various different models using a small dataset which we filter out 10 classes and 100,000 data points from the main dataset. Above are the results. As expected, since our features are nominal features with no linearity within the features, K-Nearest Neighbor is not working well. We also tried Logistic Regression briefly which we didn't put in this notebook and that didn't work well either. Random Forest Classifier also didn't work too well as we had to do OneHotEncoding to the features to avoid RandomForestClassifier from treating them like continuous features. However due to the high cardinality of our features, OneHotEncoding transform the original 14 features that we have to over 25,000 features which bogged the model and made the model needed a certain degree of depth to be effective. Notice that the trainset and holdset scores are close, showing no sign of overfit even though we already had a tree depth of 15. We probably had to relax the tree depth to deeper for better results. However that come at a computational trade off which we showed that is very expensive as we didn't even managed to run the model with the full dataset.

This left us with our 2 best model using the CategoricalNB and TensorFlow Recommendation System. We tried using CategoricalNB with SMOTEN which didn't return to us significantly better results. When we tried to run SMOTEN on 100 classes, our system ran out of memory. As such we decided to test out the full dataset using our CategoricalNB (without SMOTE) model and our Advanced TensorFlow Recommendation System Model. Their results are as below.

|                     | Baseline | CategoricalNB (without SMOTE) | Advanced TFRS |
|---------------------|----------|-------------------------------|---------------|
| Trainset MAPK Score | 0.0725   | 0.4630                        | 0.8071        |
| Holdset MAPK Score  | 0.0725   | 0.3440                        | 0.3050        |
| Kaggle Score        | 0.0700   | 0.2941                        | 0.2194        |

Our best model is still CategoricalNB with a kaggle score of 0.2941 while our TFRS model achieved a score of 0.2194. Please note that work remains to be done for both our models especially our TFRS model which is still overfitted.

Possible Further Improvement for the CategoricalNB Model:

- We can try balancing the different classes more by using an undersampling for some of the classes and oversampling for some of the classes within memory allowance.

- We can also explore using partial_fit in CategoricalNB to manage the memory usage. This will require the use of a custom pipeline.

Possible Further Improvement for the TensorFlow Recommendation System Model:
    
- Finetune the model to address overfitting using early stopping or regularization

- Build a multitask TFRS model which include a ranking task.

That saying, we like the potential of our TFRS model. Despite us not yet having prevented it from overfitting, it already achieve a score close to our best model, the CategoricalNB model. Also there is ways that we can improve the model compared to the CategoricalNB Model.

We also like that the model can handle new categories with ease which is a challenge for all our other ML models including the CategoricalNB where we had to write a preprocessing step to categorical all new categories as 'others. Furthermore we can train this model by batches which will sort the issue of out of memory when training with huge datasets. This also means that we do not have to retrain the model when we have need data and we can keep improving the model when we have new data. In addition, the model is one of the models trained the fastest.

This model also have the flexibility to be reconfigured for different requirement. For example as mentioned earlier, our model is configured to recommend hotel cluster based on the user past preferences and searched destination. We can also reconfigure the model to recommend destinations and hotel cluster based on user past preferences by putting destination under the Candidate Model and using the destination dataset to find similarities between destinations.

In conclusion, we are satifised with the performance of our top 2 models especially the TFRS model which achieved a kaggle score of 0.2194 and beat the baseline score of 0.07. At the same time, the models can be further improved to achieve better scores which match the top scores in Kaggle.

---
## 7. References and Data Sources

1) https://towardsdatascience.com/how-i-was-using-naive-bayes-incorrectly-till-now-part-1-4ed2a7e2212b

2) https://towardsdatascience.com/naive-bayes-classifier-how-to-successfully-use-it-in-python-ecf76a995069

3) https://www.tensorflow.org/recommenders

4) https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159

5) https://towardsdatascience.com/how-to-encode-categorical-data-d44dde313131

6) https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd

7) https://www.kaggle.com/c/expedia-hotel-recommendations/overview

8) https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
