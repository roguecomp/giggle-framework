from gframe import gframe
import pandas as pd
import numpy as np
import numba

# %%
@numba.jit(forceobj=True, parallel=True)
def custom(predicted, actual):
    guessed_correctly = np.sum(np.equal(np.array(np.round(predicted)), np.array(actual)))
    return "{}/{} : {:.5f}".format(guessed_correctly, len(predicted), guessed_correctly / len(actual))
temp = custom([0], [0])
# %%
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df=pd.read_csv("../data/train.csv", low_memory=False)
from sklearn.metrics import mean_squared_error
gframe(df) \
    .fillna(median=True, categories = ['Embarked', 'Sex'], drop = False) \
    .train(models=[LinearRegression(), RandomForestClassifier(n_jobs=8)],
           Y = 'Survived',
           x = ['Age', 'Pclass', 'male', 'female', 'S', 'C', 'Q'],
           metrics = [mean_squared_error, custom])
# %%

