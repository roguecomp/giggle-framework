#from gframe import *
from gframe import gframe
import numpy as np
import pandas as pd

# %%
def custom(predicted, actual):
    guessed_correctly = sum(np.array(np.round(predicted)) == np.array(actual))
    return "{}/{} : {:.5f}".format(guessed_correctly, len(predicted), guessed_correctly / len(predicted))
# %%
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df=pd.read_csv("../data/train.csv", low_memory=False)
from sklearn.metrics import mean_squared_error
gframe(df) \
    .fillna(median=True, categories = ['Embarked', 'Sex'], drop = False) \
    .train(models=[LinearRegression(), RandomForestClassifier()],
           Y = 'Survived',
           x = ['Age', 'Pclass', 'male', 'female', 'S', 'C', 'Q'],
           metrics = [mean_squared_error, custom])
# %%

