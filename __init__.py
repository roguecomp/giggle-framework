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
    
    def cat_to_num(col : str) -> list:
        categories = self.df[col].unique()
        features = []
        for cat in categories:
            binary = (self.df == cat)
            features.append(binary.astype("int"))
        return features
    
    def fill_na(self, median : bool = True, mode : bool = True, mean : bool = True):
        for col in self.df:
            if df[col].dtype == float:
                if mean:
                    df[col].fillna(df[col].mean(), inplace = True)
                else:
                    df[col].fillna(-99999, inplace = True)
                    
            elif df[col].dtype == int:
                if median:
                    df[col].fillna(df[col].median(), inplace = True)
                else:
                    df[col].fillna(-1, inplace = True)

            else:
                if mode:
                    df[col].fillna(df[col].mode(), inplace = True)
                else:
                    df[col].fillna(-1, inplace = True)
                    
                
                continue 

    
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
        n_splits = 5) -> tuple:
        
        train, val = self.cfolds(y=y, n_splits=n_splits)
        
        for model in models:
            model.fit(X=x, y=y)
        
        
        
        pass
        
# %%

gframe(df).fill_na()
# %%

# %%
