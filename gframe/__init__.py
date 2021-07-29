# %%
import pandas as pd
import sklearn.model_selection as model_selection
from typing import Union
import numpy as np
import time
# %%
class gframe:
    def __init__(self, df : pd.DataFrame) -> None:
        try:
            assert type(df) == pd.DataFrame
            
        except AssertionError:
            raise TypeError('Received type {}, Expected {}'.format(type(df), pd.DataFrame))

        self.df = df.copy()

    def __len__(self) -> int:
        """Returns number of rows in DataFrame

        Returns:
            int: Number of rows
        """
        return self.df.index.stop
    
    def shape(self) -> tuple:
        """Returns the shape of the DataFrame

        Returns:
            tuple: Returns (row x column) shape of DataFrame
        """
        return self.df.shape
    
    def cat_to_num(self, col : str) -> None:
        """Changes categories to binary columns

        Args:
            col (str): Column in DataFrame
            drop (bool, optional): Should it drop original column. Defaults to False.
        """
        categories = self.df[col].dropna().unique()
        features = []
        for cat in categories:
            binary = (self.df[col] == cat)
            self.df[cat] = binary.astype("int")
    
    def fillna(self, median : bool = False, mean : bool = False,
               categories : Union[str, list] = [], drop : bool = False ) -> pd.DataFrame:
        """fills null values in data frame

        Args:
            median (bool, optional): fill null values with median. Defaults to False.
            mean (bool, optional): fill null values with mean. Defaults to False.
            categories (Union[str, list], optional): changes columns to boolean columns. Defaults to [].
            drop (bool, optional): drops original column after making them boolean. Defaults to False.

        Raises:
            KeyError: If passed in column from list is not in DataFrame exception in raised
        """
        
        for col in self.df:
            if self.df[col].dtype in [float, int]:
                if mean:
                    self.df[col].fillna(self.df[col].mean(), inplace = True)
                elif median:
                    self.df[col].fillna(self.df[col].median(), inplace = True)
                else:
                    self.df[col].fillna(-99999, inplace = True)
                    
            else:
                try:
                    if type(categories) == list:
                        for cat in categories:
                            self.cat_to_num(cat)
                    else:
                        self.cat_to_num(categories)
                 
                except KeyError:
                    raise KeyError(' "{}" is not a column in the DataFrame'.format(cat))

        if drop: self.df.drop(categories, axis=1, inplace=True)
        
        return gframe(self.df)
    
    def cfolds(self, y : str = 'target', n_splits : int = 5 ) -> tuple:
        """Creates validation and train split using sklearn kfolds. 
        
        Args:
        -----
            y (str):
                The column that you are trying to predict. This column
                name will be useful when trying to split the DataFrame
                using kfolds.
                
            n_splits (int):
                How many splits to apply to the DataFrame to obtain the 
                validation set. validation set is (1/n_splits) of the 
                original DataFrame. Defaults to 5.

        Raises:
            KeyError: Column specified in target is not in DataFrame

        Returns:
            tuple (train_DataFrame, val_DataFrame): Train and Val split DataFrame
        """
        
        if y not in self.df:
            raise KeyError('{} is not in the DataFrame, did you forget to set y=column '.format(y))
        
        self.df['kfold'] = -1
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=self.df[y].values)):
            self.df.loc[val_idx, 'kfold'] = fold

        return (self.df.loc[self.df['kfold'] != 0], self.df.loc[self.df['kfold'] == 0])
        
    def train(self, models : list, Y : Union[str, list],
              x : Union[str, list], metrics : list = []) -> tuple:
        """Trains given list of models while using given columns. 

        Args:
        -----
            models (list):
                this list contains the various sckit learn models that
                are to be used to train
                
            y (Union[str, list]):
                Specifies the name on the column in the DataFrame that
                the model is supposed to PREDICT.
                
            x (Union[str, list]):
                Specifies the names of the columns that are USED to PREDICT
                the target column.

            metrics (list, optional):
                Specifies the metric that can be used to evaluate model performance

        Returns:
            tuple: [description]
        """
        
        # terminal colors
        red = "\u001b[31;1m"
        green = "\u001b[32m"
        yellow = "\033[;33m"
        blue = "\033[;34m"
        end = "\033[0m"
        
        for model in models:
            print("[MODEL] {}".format(str(model)))
            print()
            for fold in range(2, 15):
                scores = []
                since = time.time()
                train, val = self.cfolds(y=Y, n_splits=fold)
                
                # Initialize X and y values in right format
                X = np.array(train[x])
                y = np.array(train[Y])
                
                model.fit(X=X, y=y)
                
                for metric in metrics:
                    # interpolation in order to spot overfitting in models
                    score = metric(model.predict(X), y)
                    scores.append(score)
                
                X = np.array(val[x])
                y = np.array(val[Y])
                
                now = time.time()
                
                for i, metric in enumerate(metrics):
                    score = metric(model.predict(X), y)
                    if type(scores[i]) != str: scores[i] = np.round(scores[i], 5) 
                    if type(score) != str: score = np.round(score, 5) 
                    
                    # A red colour output here shows the training set prediction 
                    # results whereas a green colour output here shows the validation
                    # set prediction results
                    print(f"[ {yellow}{metric.__name__}{end} ]: {red}{scores[i]},{end} {green}[V] {score}{end}")
                    
                print(f"{blue} ===> Fold:{end} {fold}/14 || {blue}Time taken:{end} {now - since:.3f} s")
                
                print()
            print('=' * 50)
        
        self.models = models
