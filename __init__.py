# %%
import pandas as pd
import sklearn.model_selection as model_selection
from typing import Union

# %%
df=pd.read_csv("data/train.csv")

# %%
class gframe:
    def __init__(self, df : pd.DataFrame) -> None:
        try:
            assert type(df) == pd.DataFrame
            
        except AssertionError:
            raise TypeError('Received type {}, Expected {}'.format(type(df), pd.DataFrame))

        self.df = df

    def __len__(self) -> int:
        """Returns number of rows in DataFrame

        Returns:
            int: Number of rows
        """
        return df.index.stop
    
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
    
    def fillna(self, median : bool = True, mean : bool = True, categories : Union[str, list] = [], drop : bool = False) -> None:
        for col in self.df:
            if self.df[col].dtype == float:
                if mean:
                    self.df[col].fillna(self.df[col].mean(), inplace = True)
                else:
                    self.df[col].fillna(-99999, inplace = True)
                    
            elif self.df[col].dtype == int:
                if median:
                    self.df[col].fillna(self.df[col].median(), inplace = True)
                else:
                    self.df[col].fillna(-1, inplace = True)

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
    
    def cfolds(self, y : str = 'target', n_splits : int = 5) -> tuple:
        
        """Creates validation and train split using folds

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
        
    def train(
        self,
        models : list,
        y : Union[str, list] = 'target',
        x : Union[str, list] = 'Age',
        n_splits = 5 ) -> tuple:
        
        train, val = self.cfolds(y=y, n_splits=n_splits)
        print(train)
        
        # for model in models:
        #     model.fit(X=x, y=y)
        
        pass
        
# %%
from sklearn.linear_model import LinearRegression

a = gframe(df)
a.fillna(categories = ['Embarked'], drop = True)
# a.train(models=[LinearRegression()], y='Survived', x='Age')
# %%
a.df
# %%
