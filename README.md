# shipment-analytics-prediction-deep-learning

### The Repository contains nine solid projects placed in different folders on shipment and logistics.
### The projects span across the different fields of Data science, including: Data Analysis, Machine Learning, and Deep Learning 
### There are projects that use different deep learning techniques like Recurrent Neural Network (RNN), Deep Neural Network (DNN), and Convolutional Neural Network(CNN) to predict disruptions, anomalies, fleet maintenance, among others in the Shipment and Logistics industry.


## Project 1: Data Preprocessing.

### **Objective:**:

* #### Clean and preprocess a dataset for shipment times and routes.

### **Some methods to apply**:
1. Load the dataset from a CSV file.
2. Handle missing values appropriately.
3. Encode categorical variables.
4. Normalize numerical features.

### Methods Used for Project 1:

* Loaded dataset with 25 features and 25k records
* Verified no missing values or duplicates
* Selected key features: Routes, Shipping times, Distance, etc.
* Encoded categorical 'Routes' with LabelEncoder
* Normalized numerical features with StandardScaler

### Results:

#### Successfully completed preprocessing with:

* Clean dataset (no missing values/duplicates)
* Normalized column names
* Properly encoded categorical variable
* Appropriately scaled numerical features

### Possible Improvement(s):

* Implement ColumnTransformer for a unified preprocessing pipeline
* Add feature correlation analysis before selection
* Use OneHotEncoder instead of LabelEncoder for nominal data
* Explore more/fewer features.

## Project 2:

### **Objective:**:

* #### Visaul Representation and Interpretion of Shipping Logistics Data

  ### Methods Used for Project 2:
* Revealed Trends, Bottlenecks & Regional Performance.
* Extracted and communicated Valuable insights.
* Data Storytelling.

## Project 3:  Model Training with scikit-learn

### **Objective:**:

* #### Train a machine learning model to predict shipment times.

### **Some methods to apply**:

1. Split the dataset into training and testing sets.
2. Train a regression model.
3. Evaluate the model's performance.

### Methods Used for Project 3:

* Split data (75% train, 25% test)
* Trained LinearRegression, RandomForestRegressor, SVR models, XGBRegressor, and LGBMRegressor
* Evaluated with each model's MSE

### Results: Random Forest achieved MSE: 0.283

* #### Strong performance achieved.
* #### The Gradient Boosters are currently the best performer without hyperparameter tuning on the other models
* #### The low MSE for Random Forest and the gradient boosting regressors reflects a high level of predictive accuracy.
* The LGBMRegressor and XGBRegressor achieved the lowest errors (MSE = 0.2532 and 0.2549), indicating they best capture complex interactions.
* RandomForestRegressor and SVR followed with moderate errors (MSE = 0.2831 and 0.3518), showing some non-linear patterns learned.
* Linear Regression had the highest error (MSE = 0.7589), suggesting it underfits non-linear relationships.


### Possible methods for improvement:

* I strongly believe the selected features are impacting the results more than the algorithms.
* Random Forest model's performance improves with better feature selection
* I may try polynomial features for a non-linear relationship.
* Use cross-validation for more reliable metrics.
* Apply hyperparameter tuning using GridSearchCV or RandomSearchCV


## Project 4: Neural Network with TensorFlow

### **Objective:**

* #### Build a neural network to predict shipment times.

### **Some methods to apply**:
1. Define a neural network architecture.
2. Train the neural network.
3. Evaluate the model's performance.

### Methods Used:

* Built 3-layer Sequential model (64-32-1 neurons)
* Used Adam optimizer and MSE loss
* Trained for 50 epochs (batch_size=32)


### Results: Achieved MSE: 0.276

* The result is Comparable to Random Forest regression and gradient boosters MSE in Task 2
* Slight improvement over the Random Forest model
* Performed slightly lower than the gradient boosters
* Potential underfitting observed

### Improvement Suggestions:

* I may need to add dropout layers for regularization
* Implement early stopping to prevent overfitting
* And experiment with different activation functions

## Project 5: Predicting Supply Chain Disruptions with Pytorch

### **Objective:**

* #### Build a model to predict supply chain disruptions based on external factors.

### **Some methods to apply**:
    1. Load and preprocess external data (e.g, weather patterns).
    2. Train a classification model using PyTorch.
    3. Evaluate the model's accuracy.


### Methods Used for Project 5:

* #### I obtained my weather data by finding four separate real-world weather datasets for the four locations (Delhi, Bangalore, Chennai, and Mumbai) on my extrapolated shipment data.
* #### I focused on Locations when sourcing for the data to have a common column to merge my shipment data on.
* #### The code I used for this whole preprocessing and aggregation is in a separate notebook to avoid clustering. However, I save the dataset as a CSV file named weather_df.csv, and I will be reading it here for a merge
* I extracted the years I wanted (2012 to 2022).
* I concatenated the four datasets and sorted the concatenated dataset by date to shuffle the location column.
* I also added a weather column to the weather dataset based on a temperature threshold to categorize water conditions.
* Created binary 'disruption' target variable
* Built 3-layer PyTorch classifier
* Achieved 82.85% accuracy

### Results:

* Decent but improvable performance
* Accuracy is acceptable for the initial model
* Potential class imbalance not addressed

### Possible improvement methods:

* I may add class weighting for imbalance
* Include some temporal features
* Use other evaluation metrics like ROC-AUC for better evaluation

## Project 6: Time Series Forecasting

### **Objective:**

* #### Build a model to forecast shipment times based on historical data.

### **Some methods to apply**:

1. Prepare the time data.
2. Define and train a recurrent neural network (RNN).
3. Evaluate the model's performance.

### Methods Used for Time Series Forecasting Project 6:

* Created time-step sequences (window=10)
* Built an LSTM model with 50 units
* Trained for 50 epochs

### Results: MSE 0.4691

* Decent performance

### Critical Improvements Needed:

* Maybe include multivariate features (weather, routes)
* Add seasonal decomposition
* Implement sequence-to-sequence architecture
* Use differencing for stationarity

## Project 7: Image Classification for Package Inspection.

### **Objective:**

* #### Build and evaluate a convolutional neural network (CNN) to classify images of packages.

### **Some methods to apply**:

1. Load and preprocess the image dataset.
2. Define and train a CNN.
3. Evaluate the model's performance.

### Methods Used for Computer Vision in Project 7:

* Sourced images from many sources like Bing, Kaggle, etc
* Finally got some unclassified dataset from RoboFlow
* I classified the dataset myself for the train, validation, and test
* I used ImageDataGenerator for augmentation
* I built a 3-conv layer CNN
* Trained for 20 epochs

### Results: Achieved 85.9% validation accuracy.

* #### Good Performance
* Reasonable accuracy for binary classification
* I applied proper data augmentation.

### Improvement Suggestions:

* I will implement transfer learning (ResNet/VGG),
* Add batch normalization layers and
* Use learning rate scheduling


## Project 8: Anomaly Detection in Shipment Data.

### **Objective:**

* #### Build an autoencoder to detect anomalies in shipment data.

### **Some methods to apply**:

1. Prepare the dataset for anomaly detection.
2. Define and train an autoencoder.
3. Identify anomalies based on reconstrucƟon error.

### Methods Used Anomaly Detection Project 8:

* I built an incomplete autoencoder (2-neuron bottleneck)
* I used the reconstruction error threshold (95th percentile)
* Identified 1229 anomalies

### Results:

* Functional but simplistic
* Basic architecture limits feature extraction
* Arbitrary threshold selection

### Possible improvement methods:

* Use a variational autoencoder (VAE)
* Implement isolation forests
* Add domain-based anomaly rules

## Project 9: Predictive Maintenance for Fleet Management

### **Objective:**

* #### Build a model to predict maintenance needs for a fleet of vehicles.

### **Some methods to apply**:

1. Load and Preprocess the maintenance dataset.
2. Define and train a predictive model.
3. Evaluate the model's performance.

### Methods Used for Predictive Maintenance Project 9:

* I preprocessed the data
* I selected mileage, age, and maintenance cost as specified
* Built 3-layer DNN (64-32-1)
* Achieved MSE: 0.8079

### Results:

* Suboptimal model performance

### Improvement Suggestions:

* Add survival analysis models
* Incorporate sensor data (engine temp)
* Implement remaining useful life (RUL) estimation
