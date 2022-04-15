# Disaster Response Pipeline

## Project Description
Our goal is to build a model that classifies messages sent during disasters. There are 36 pre-defined categories in this model, and examples include aid-related messages, medical help messages, and search and rescue messages.
We will construct an ETL and Machine Learning pipeline to facilitate the delivery of these messages to the appropriate disaster relief agency. As a message can belong to more than one category, so this is also a multi-label classification task. We will work with a dataset provided by Figure Eight that conatins real-life messages sent during disaster events.
Lastly, this project includes a web app where you can input a message and get classification results.

![image](https://user-images.githubusercontent.com/10460709/163563697-09084c65-3b19-48ea-8e50-2f6e092c47ce.png)

## File Description
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- README

## Libraries used:
- numpy
- pandas
- sqlalchemy
- nltk
- re
- pickle
- sklearn
- plotly
- flask

## File Descriptions
App folder including the templates folder and "run.py" for the web application
Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
README file
Preparation folder containing 6 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)

## Instructions
- Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    - To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py
- Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements
Many thanks to Figure-8 for making this available to Udacity for training purposes. Special thanks to udacity for the training. Feel free to utilize the contents of this while citing me, udacity, and/or figure-8 accordingly
