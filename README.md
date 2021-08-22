
# Domestic Flight Fare Predictor

![App Screenshot](https://github.com/bharathngowda/machine_learning_domestic_flight_price_prediction/blob/main/Breeze%20Airways_166655077_303814634409055_8038496796049085212_n.jpeg)

### Table of Contents

1. [Problem Statement](#Problem-Statement)
2. [Data Pre-Processing](#Data-Pre-Processing)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Feature Engineering](#Feature-Engineering)
5. [Model Training](#Model-Building)
6. [Model Selection](#Model-Selection)
7. [Model Evaluation](#Model-Evaluation)
8. [Dependencies](#Dependencies)
9. [Installation](#Installation)

### Problem Statement

The goal of the project is to use the train set to train a regression model and predict the flight fares for the test set.

**Quick Start:** [View](https://github.com/bharathngowda/machine_learning_domestic_flight_price_prediction/blob/main/Domestic%20Flight%20Price%20Predictor.ipynb) a static version of the notebook in the comfort of your own web browser.

### Data Pre-Processing

- Loaded the train and test data
- Checking for null values 
- Dropping null values as there are only 2 null values


### Exploratory Data Analysis

- **Correlation Plot** to determine if there is a linear relationship between the 2 variables and the strength of the relationship
- **Pair Plot**  to see both distribution of single variables and relationships between two variables
- Plots to understand which airlines are popular, popular source and destinations, travel routes for different airlines,number of airlines with no stops, 1stops etc.., flight durations, price range of different airlines, price range for source and destination.


### Feature Engineering

New features created are - 
* Year - by extracting year from date of journey
* Month - by extracting month from date of journey
* Day - by extracting day from date of journey
* Converted Stops from categorical to numerical
* Converted Arrival and departure time to datetime object
* Extracted arrival hour and arrival min from arrival time
* Extracted departure hour and departure min from departure time

### Model Training

Models used for the training the dataset are - 

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [DicesionTree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

### Model Selection

I have used [r2 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) as my scorer and used [k-fold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
to select the model with highest **'r2 score'**.

### Hyperparameter Turning 

I have performed Hyperparameter Tuning on the selected model using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and selected the best parameters to obtain the final model.


### Model Evaluation

I fit the final model on the train data and predicted the flight fare for the test data and obtained the below result-

| Metric    | Score    |
| :-------- | :------- |
| MAE	|1.157466e+03
| MSE	|3.727888e+06
| RMSE	|1.930774e+03
| MSLE	|3.126863e-02
| R^2	|8.115039e-01

### Dependencies
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

### Installation

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/bharathngowda/machine_learning_domestic_flight_price_prediction/archive/refs/heads/main.zip) or execute this from the terminal:
`git clone https://github.com/bharathngowda/machine_learning_domestic_flight_price_prediction.git`

2. Install [virtualenv](http://virtualenv.readthedocs.org/en/latest/installation.html).
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with `virtualenv env`.
4. Activate the environment with `source env/bin/activate`
5. Install the required dependencies with `pip install -r requirements.txt`.
6. Execute `ipython notebook` from the command line or terminal.
7. Click on `Domestic Flight Price Predictor.ipynb` on the IPython Notebook dasboard and enjoy!
8. When you're done deactivate the virtual environment with `deactivate`.
