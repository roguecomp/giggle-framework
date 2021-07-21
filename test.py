from gframe import *

# %%
def custom(x, y):
    return "{}/{} : {}".format(sum(np.array(np.round(x)) == np.array(y)), len(x), sum(np.array(np.round(x)) == np.array(y)) / len(x))
# %%
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df=pd.read_csv("data/train.csv", low_memory=False)
from sklearn.metrics import mean_squared_error
gframe(df) \
    .fillna(median=True, categories = ['Embarked', 'Sex'], drop = False) \
    .train(models=[LinearRegression(), RandomForestClassifier()],
           Y = 'Survived',
           x = ['Age', 'Pclass', 'male', 'female', 'S', 'C', 'Q'],
           metrics = [mean_squared_error, custom])
# %%

