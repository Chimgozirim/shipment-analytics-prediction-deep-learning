# shipment-analytics-prediction-deep-learning
Data Analytics and Data Science Projects on Shipment and Logistics.


## Project 1: Data Preprocessing.

**Objective**

Clean and preprocess a dataset for shipment times and routes.

**Instructions**
1. Load the dataset from a CSV file.
2. Handle missing values appropriately.
3. Encode categorical variables.
4. Normalize numerical features.

## Methods Used for Task 1:

* Loaded dataset with 25 features and 25k records
* Verified no missing values or duplicates
* Selected key features: Routes, Shipping times, Distance
* Encoded categorical 'Routes' with LabelEncoder
* Normalized numerical features with StandardScaler

## Results:

### Successfully completed preprocessing with:

* Clean dataset (no missing values/duplicates)
* Normalized column names
* Properly encoded categorical variable
* Appropriately scaled numerical features

## Possible Improvement(s):

* Implement ColumnTransformer for a unified preprocessing pipeline
* Add feature correlation analysis before selection
* Use OneHotEncoder instead of LabelEncoder for nominal data

## Project 2

## Project 3:  Model Training with scikit-learn

**Objective**

Train a machine learning model to predict shipment times.

**Instructions**

1. Split the dataset into training and testing sets.
2. Train a regression model.
3. Evaluate the model's performance.

## Methods Used for Task 2:

* Split data (75% train, 25% test)
* Trained LinearRegression, RandomForestRegressor, and SVR models
* Evaluated with each model's MSE

## Results: Random Forest achieved MSE: 0.283

### Strong performance achieved:

* Random Forest Regressor achieved the lowest error (MSE = 0.2831), indicating it best captures complex interactions.
* SVR followed with moderate error (MSE = 0.3518), showing some non-linear patterns learned.
* Linear Regression had the highest error (MSE = 0.7589), suggesting it underfits non-linear relationships.

* #### The low MSE for Random Forest reflects a high level of predictive accuracy.

## Possible methods for improvement:

* I strongly believe the selected features are impacting the results more than the algorithms.
* Random Forest model's performance improves with better feature selection
* I may try polynomial features for a non-linear relationship.
* Use cross-validation for more reliable metrics.

## The GradientBoosters are currently the best performer without hyperparameter tuning on the other models


## Project 4: Neural Network with TensorFlow

**Objective:**

Build a neural network to predict shipment times.

**Instructions**
1. Define a neural network architecture.
2. Train the neural network.
3. Evaluate the model's performance.

## Methods Used:

* Built 3-layer Sequential model (64-32-1 neurons)
* Used Adam optimizer and MSE loss
* Trained for 50 epochs (batch_size=32)


## Results: Achieved MSE: 0.276

* The result is Comparable to Random Forest regression and gradient boosters MSE in Task 2
* Slight improvement over the Random Forest model
* Performed slightly lower than the gradient boosters
* Potential underfitting observed

## Improvement Suggestions:

* I may need to add dropout layers for regularization
* Implement early stopping to prevent overfitting
* And experiment with different activation functions

## Project 5: Predicting Supply Chain Disruptions with Pytorch

### Objective:
#### Build a model to predict supply chain disruptions based on external factors.

### Instructions:
    1. Load and preprocess external data (e.g, weather patterns).
    2. Train a classification model using PyTorch.
    3. Evaluate the model's accuracy.

#### I obtained my weather data by finding separate weather data for four locations on my extrapolated shipment data (Delhi, Bangalore, Chennai, and Mumbai). I extracted the years I wanted (2012 to 2022), then I concatenated the 4 dataset and sort it by dates to shuffle the location column.

#### I focused on Locations when sourcing for the data to have a common column to merge my shipment data on.

### The code I used for this whole preprocessing and aggregation is in a separate notebook to avoid clustering. However, I save the dataset as a CSV file named weather_df.csv, and I will be reading it here for a merge

### I also added a weather column to the weather dataset based on the temperature threshold to categorize water conditions.


## Methods Used for Task 4: Disruption Prediction in Task 4: 

* Concatenated 4 real-world weather datasets eligible for merging with the shipment data based on location.
* Merged weather data with shipment records
* Created binary 'disruption' target variable
* Built 3-layer PyTorch classifier
* Achieved 82.85% accuracy

## Results:

* Decent but improvable performance
* Accuracy is acceptable for the initial model
* Potential class imbalance not addressed

## Possible improvement methods:

* I may add class weighting for imbalance
* Include some temporal features
* Use other evaluation metrics like ROC-AUC for better evaluation

## Task 5: Time Series Forecasting
**Objectives**

Build a model to forecast shipment times based on historical data.

**Instructions**

1. Prepare the time data.
2. Define and train a recurrent neural network (RNN).
3. Evaluate the model's performance.

## Methods Used for Time Series Forecasting Task 5:

* Created time-step sequences (window=10)
* Built LSTM model with 50 units
* Trained for 50 epochs

## Results: MSE 0.4691

* Decent performance

## Critical Improvements Needed:

* Maybe include multivariate features (weather, routes)
* Add seasonal decomposition
* Implement sequence-to-sequence architecture
* Use differencing for stationarity

## Methods Used for Computer Vision in Task 6:

* Sourced images from many sources like Bing, Kaggle, etc
* Finally got some unclassified dataset from RoboFlow
* I classified the dataset myself for the train, validation, and test
* I used ImageDataGenerator for augmentation
* I built a 3-conv layer CNN
* Trained for 20 epochs
* Achieved 85.9% validation accuracy

## Results:

### Good Performance

* Reasonable accuracy for binary classification
* I applied proper data augmentation.

## Improvement Suggestions:

* I will implement transfer learning (ResNet/VGG),
* Add batch normalization layers and
* Use learning rate scheduling

## Methods Used Anomaly Detection Task 7:

* I built an incomplete autoencoder (2-neuron bottleneck)
* I used the reconstruction error threshold (95th percentile)
* Identified 1229 anomalies

## Results:

* Functional but simplistic
* Basic architecture limits feature extraction
* Arbitrary threshold selection

## Possible improvement methods:

* Use a variational autoencoder (VAE)
* Implement isolation forests
* Add domain-based anomaly rules

## Task 8: Predictive Maintenance for Fleet Management

### **Objective**

#### Build a model to predict maintenance needs for a fleet of vehicles.

### **Instructions**

1. Load and Preprocess the maintenance dataset.
2. Define and train a predictive model.
3. Evaluate the model's performance.

## Methods Used for Predictive Maintenance Task 8:

* I preprocessed the data
* I selected mileage, age, and maintenance cost as specified
* Built 3-layer DNN (64-32-1)
* Achieved MSE: 0.8079

## Results:

* Suboptimal model performance

## Improvement Suggestions:

* Add survival analysis models
* Incorporate sensor data (engine temp)
* Implement remaining useful life (RUL) estimation
