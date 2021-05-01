# Flask-Bike-Rental-Predictor

### This is a repo for Duke MIDS cloud computing course final project.

This project is to build a build a containerized machine learning prediction model and deploy it in Google Cloud Platform. The data used in this project is from Kaggle, [check out the competition here.](https://www.kaggle.com/c/bike-sharing-demand)

![App GUI](https://user-images.githubusercontent.com/56423760/116678129-9359cd00-a9db-11eb-83c6-7c2f777ae200.png)
### Data Description

| Variable   | Type                                                                                                                                                                                                                                                           |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| season     | 1 = spring, 2 = summer, 3 = fall, 4 = winter                                                                                                                                                                                                                   |
| holiday    | 1 = holiday, 0 = non-holiday                                                                                                                                                                                                                                   |
| workingday | 1 = workingday, 0 = non-workingday                                                                                                                                                                                                                             |
| weather    | 1 : clear, few clouds, partly cloudy 2 : mist + cloudy, mist + broken clouds, most + few clouds, mist 3 : light snow, light rain + thunderstorm + scattered clouds, light rain + scatterd clouds 4 : heavy rain + ice pallet + thunderstorm + mist, snow + fog |
| temp       | temperature in Celsuis                                                                                                                                                                                                                                         |
| atemp      | "feels like" temperature in Celsius                                                                                                                                                                                                                            |
| humidity   | relative humidity                                                                                                                                                                                                                                              |
| windspeed  | wind speed                                                                                                                                                                                                                                                     |
| casual     | number of non-registered user rentals initiated                                                                                                                                                                                                                |
| registered | number of registered user rentals initiated                                                                                                                                                                                                                    |
| count      | number of total rentals                                                                                                                                                                                                                                        |
| year       | The year of the datetime.                                                                                                                                                                                                                                      |
| month      | The month as January=1, December=12.                                                                                                                                                                                                                           |
| day        | The day of the datetime.                                                                                                                                                                                                                                       |
| dayofweek  | The day of the week with Monday=0, Sunday=6.                                                                                                                                                                                                                   |
| hour       | The hours of the datetime.                                                                                                                                                                                                                                     |

### Model Selection

A few regression models were applied to this dataset: linear regression, decision tree, random forest, and XGBoost. XGBoost was selected in the model deployment section due to the best model performance. (check model_training.ipynb for more details of data cleaning, data visualization and model training)

```
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

columns = ['count', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'dayofweek','hour']
train_file = './data/bike_train.csv'
validation_file = './data/bike_validation.csv'
# Specify the column names as the file does not have column header
df_train = pd.read_csv(train_file,names=columns)
df_validation = pd.read_csv(validation_file,names=columns)

X_train = df_train.iloc[:,1:] # Features: 1st column onwards 
y_train = df_train.iloc[:,0].ravel() # Target: 0th column

X_validation = df_validation.iloc[:,1:]
y_validation = df_validation.iloc[:,0].ravel()

regressor = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 50)

regressor.fit(X_train, y_train)

# Saving model using pickle
regressor.save_model('model.json')


print('done')
```

### Model Deployment

The model hosted on Google Cloud Platform using Flask app.

Main steps are:

1. Create new project on GCP, and verify project is working (verify you have the correct project, and switch to it if necessary)

```
gcloud projects describe $PROJECT_ID
gcloud config set project $PROJECT_ID
```

2. Create app engine app

```
gcloud app create
```

3. Clone this repo and cd into it

```
git clone https://github.com/jingjing-shi/Flask-Bike-Rental-Predictor.git
cd Flask-Bike-Rental-Predictor
```

4. Create a virtual environment

```
virtualenv --python $(which python3) venv
source venv/bin/activate
```

5. Install all required packages and run locally
```
make install
python main.py 
```

6. Deploy the app and now you will be able to access it [here](https://bike-rental-312210.ue.r.appspot.com)

```
gcloud app deploy
```

### Continuous Delivery

* Create a new build trigger and connect to this github repo
* Make sure you enable required APIs and set necessary permissions to service account
* Set up the cloudbuild.yaml file
* Now you will able to track code changes in this repo from the build triggers page

![Example of Continuour Delivery Page](https://user-images.githubusercontent.com/56423760/116678445-e59aee00-a9db-11eb-91b0-3518f19feed8.png)
### Load Testing

*Tested with Locust*

```

import time
from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    #wait_time = between(1, 2.5)

    @task
    def submitForm(self):
        self.client.post("/predict", 		 
         {"season":"1",
        "holiday":"0",
        "workingday":"1",
        "weather":"3",
        "temp":"28.2",
        "atemp":"34",
        "humidity":"80",
        "windspeed":"8",
        "year":"2001",
        "month":"9",
        "day":"20",
        "dayofweek":"7",
        "hour":"20"})

```

![Load Test Result](https://user-images.githubusercontent.com/56423760/116766855-47506c00-aa5f-11eb-96eb-ff0039ca2e60.png)

*I think this result will be higher if we don't have to access GCP server via a VPN*
