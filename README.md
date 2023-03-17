REQUEST - BEFORE READING, I NEED HELP ON THE BUILDSPEC.YML FILE. I WANT TO DEPLOY THIS FLASK WEB APP FOR THE DL MODEL USING AWS CODEPIPELINE TO AWS ELASTIC BEANSTALK. AND FOR 
SOME REASON I GET A SUCCESSFUL BUILD BUT DEPLOYMENT FAILS EVERYTIME. IF YOU HAVE A SOLUTION FOR THIS OR A BETTER BUILDSPEC.YML AND REQUIREMENTS.TEXT FILE, I REQUEST TO
GENERATE A PULL REQUEST!
 
Namaste,

In this application, I have created a Deep learning model to predict the stock prices of the Indian listed company Vedanta Ltd.
I have used LSTM to make the model. The dataset was collected from Yahoo Finance and is uploaded on my Kaggle account at :- https://www.kaggle.com/datasets/venubanaras/vedanta-nse-price-history

To create a CSV file using Yahoo Finance, simply go to the website, enter the stock name, select historical data
, select the time period (I selected MAX, corresponding to the company's listing date), select historical prices and the frequency (i.e. daily,weekly or monthly)
and then download. It will automatically create a .csv file for you to work upon.


The main purpose of this app was to create a deployment of a DL app and the second purpose was to learn Time Series forecasting and also to learn creating datasets
for DL projects.
I have used TensorFlow to create the models and Flask to create a deployment version for the same. The detailed explanation for the code is in the files itself.

To check out my notebook at Kaggle follow :- https://www.kaggle.com/code/venubanaras/vedanta-full-pred

Do follow me on Kaggle and give upvotes to the dataset and the notebook, if you like them.
