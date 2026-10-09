Explainable AI for Autonomous Vehicle Safety


Project Overview

For this project, I am looking at whether machine learning can help identify higher-risk autonomous vehicle safety events using historical disengagement and collision reports. I am using data from the California DMV that contains information about autonomous vehicle disengagements, testing mileage, and reported collisions. I want to understand the patterns in this data and see whether a machine learning model can help identify events that may have a higher safety risk.The goal is not only to make a prediction but also to understand what factors may contribute to the prediction.

Research Question
Can machine learning predict autonomous vehicle safety risk using historical disengagement and crash reports, and what factors contribute most to that risk?


Data Sources
For this project, I am using publicly available autonomous vehicle data from the California DMV.
I am using three datasets:
1. Autonomous Vehicle Disengagement Reports — This dataset contains information about disengagement events, including manufacturer, date, location, who initiated the disengagement, and a description of the event.
2. Autonomous Mileage Reports — This dataset contains the autonomous testing mileage reported by manufacturers. I use this data to understand the disengagement counts because different manufacturers may have driven different numbers of miles.
3. Autonomous Vehicle Collision Reports — This dataset contains information about reported collisions, including manufacturer, collision circumstances, operating mode, injury information, and collision type.
For the initial machine learning model, I use 2024 data so that the disengagement and collision events are compared over the same year.

Data Cleaning and Preparation
I first checked the datasets for missing values, duplicate records, incorrect data types, and inconsistent values. The disengagement dataset contained many completely blank rows, which I removed. I also removed completely empty columns from the mileage dataset. I converted the date column into the correct date format and converted the annual mileage into a numeric format. I also cleaned inconsistent values such as manufacturer names, locations, and disengagement initiators so that the same categories were represented consistently. Some disengagement records appeared to be duplicates. I did not remove them because the dataset does not provide a unique event ID or exact event time to confirm whether they are actual duplicates or separate disengagement events.



Exploratory Data Analysis
I performed exploratory data analysis to better understand the patterns in the disengagement, mileage, and collision data. I looked at disengagements by manufacturer, location, initiator, and month. I also compared the number of disengagements with autonomous miles driven because manufacturers have different levels of testing activity. For the collision data, I looked at collision reports by manufacturer, injury status, common collision types, vehicle operating mode, and year. Rear-end collisions were the most common among the collision types I examined. The analysis showed that event counts can vary significantly across manufacturers. However, these counts should not be directly used to compare safety because manufacturers may have different testing mileage and levels of testing activity.


Baseline Machine Learning Model
For the initial model, I used Logistic Regression as the baseline classification model.I created a target variable called Risk_Event. Collision events were assigned a value of 1 and disengagement events were assigned a value of 0. This gives the model a measurable classification target, but it should not be considered a complete measure of autonomous vehicle safety.For the baseline model, I used manufacturer and month as the input features because these were common features available across the two datasets.The 2024 data contained 1,457 disengagement events and 101 collision events. Since there were many more disengagement events than collision events, I used balanced class weights in the Logistic Regression model.


Model Results
The Logistic Regression model produced the following results:
Accuracy: 76.9%
Precision: 19.8%
Recall: 85.0%
F1-score: 32.1%

The model correctly identified 17 out of 20 collision events in the test data and missed 3 collision events. However, it also incorrectly classified 69 disengagement events as collisions. For this safety-related problem, I focused more on recall because identifying as many collision events as possible is important. The 85% recall shows that the model identified most of the collision events, while the low precision shows that it also produced many false positives. The 5-fold cross-validation produced an average recall of 86%, although the recall varied across the different folds.


Initial Findings
The initial analysis shows that there are patterns in the disengagement and collision data that can be explored using machine learning.The baseline model was able to identify most of the collision events in the test data, with a recall of 85%. However, the low precision and high number of false positives show that the model still needs improvement.The analysis also shows that raw disengagement and collision counts should not be directly used to compare the safety of different manufacturers because they have different amounts of testing mileage and testing activity.These initial results provide a good starting point for comparing additional machine learning models and exploring more useful features.


Limitations
There are several limitations in this analysis:
1. First, I am using collision events as the higher-risk class and disengagement events as the comparison class. This helps create a classification problem, but it is not a complete measure of autonomous vehicle safety.
2. The dataset also contains many more disengagement events than collision events, which creates class imbalance.
3. The baseline model uses only manufacturer and month because these are common features available across both datasets. This limits the information available to the model.
4. Manufacturers also have different amounts of testing mileage and testing activity. Therefore, the results should not be interpreted as showing that one manufacturer is safer or less safe than another.

Next Steps
In the next phase of the project, I plan to compare the Logistic Regression baseline model with other classification models such as Decision Tree, Random Forest, and Support Vector Machine (SVM). I will also explore L1 regularization for feature selection. I will investigate whether additional useful features can be added without creating data leakage. I will compare the models using precision, recall, F1-score, confusion matrices, and cross-validation. The final analysis will also focus on explainability so that I can better understand which factors contribute to higher-risk autonomous vehicle safety events.

The complete data cleaning, exploratory data analysis, visualizations, and baseline machine learning model are available in the Jupyter Notebook: AV_Safety.ipynb