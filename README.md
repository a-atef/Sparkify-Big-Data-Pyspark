# Sparkify

by: Ahmed A. Youssef

## Data Source

The data was obtained by provided by Udacity.

## Libraries Dependency
- re
- plotly
- datetime
- pandas
- Pyspark
- flask

## Project Motivation

This project focuses on analyzing big data using Pyspark. The final product is a web application for churn classification. Customers are classified into two churn groups; churned or not.

## Project Files

- ```Sparkify.ipynb```: contains all the workflow steps; EDA analysis, ETL steps and ML modeling.
- ```app\run.py```: module to deploy and run the flask app into a local server.
- ```app\templates\master.html```: display the main page of the web app. 
- ```app\templates\timeplot.html```: display timeline plots page.
- ```app\templates\distplot.html```: display distribution plots for numerical features.
- ```app\templates\traindata.html```: display training data plots after transformation.
- ```app\templates\results.html```: display the results of the best ML model (Logistic regression). 

## Analysis Workflow

### Instructions:

- Run the following command in the app's directory to deploy and fire a local web app using flask server.
```
    - ```python run.py```
    - Go to http://localhost:3001/
```

### Results

After firing the web app, you should be able to see a the main page shown below. It will show a sample view of the original dataset along with four bar plots for the categorical features.

<img src="img\main-page.png"
     alt="main page view"
     height="60%" width="60%"
     style="margin: 10px;" />

The second page `/timeplot` is the timeplot page. This page has two timeplots on; the number of new customers Sparkify enjoys each month and the number of active users each hour.

<img src="img\timeplot-page.png"
     alt="Top 10 Categories"
     height="60%" width="60%"
     style="margin: 10px;" />

The third page `/distplot` is the distribution page which contains four histrograms for the numerical features.

<img src="img\hist-page.png"
     alt="Categories Per Message"
     height="60%" width="60%"
     style="margin: 10px;" />

The fourth page `/traindata` has one table showing a sample of the clean and transformed dataset beside a heat map to show correlation between numerical features.

<img src="img\trainingdata.png"
     alt="Top keywords"
     height="60%" width="60%"
     style="margin: 10px;" />

The fifth page `/results` has three plots to demonstrate the results of the best ML model.

<img src="img/results.png"
     alt="Classify a message"
     height="60%" width="60%"
     style="margin: 10px;" />
