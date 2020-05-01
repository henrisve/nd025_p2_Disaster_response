# Disaster Response Pipeline Project

This is the second project from the Data Scientist Nanodegree Program.

It consist of 3 parts:

### ETL Pipeline 
This part takes the two csv files, clean them up and then put it into an database.\
This can be found in `data/process_data.py`\
The csv files can be found under `data/` 

### ML Pipeline
This part will take the data from the database above, and then convert this into a format where we later can train a model. The model is then saved as a pickle file.\
This can be found in `train_classifier.py`

### web app
This part will take both the data from the database, and do some nice plots. It also allows the user to send a custom string that will be classified using the trained model.\
This can be found in `app/run.py`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Required libraries
* nltk==3.4.5
* numpy==1.18.1
* pandas==1.0.1
* scikit-learn==0.22.2.post1
* SQLAlchemy==1.3.13
* Flask==1.1.1
* plotly==4.6.0
