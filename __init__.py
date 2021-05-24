# %%
import pandas as pd
import sklearn.model_selection as model_selection

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
    
    def cfolds(self, target : str = 'target', n_splits : int = 5) -> tuple:
        
        """Creates validation and train split using folds

        Raises:
            KeyError: Column specified in target is not in DataFrame

        Returns:
            tuple (train_DataFrame, val_DataFrame): Train and Val split DataFrame
        """
        
        if target not in self.df:
            raise KeyError('{} is not in the DataFrame, did you forget to set target=column '.format(target))
        
        self.df['kfold'] = -1
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.__kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)

        for fold, (train_idx, val_idx) in enumerate(self.__kf.split(X=self.df, y=self.df[target].values)):
            self.df.loc[val_idx, 'kfold'] = fold

        return (self.df.loc[self.df['kfold'] != 0], self.df.loc[self.df['kfold'] == 0])
        
        
# %%
a = gframe(df)
train, val = a.cfolds(target = 'Survived')
train
# %%
