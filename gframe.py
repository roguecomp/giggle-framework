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
    
    def cat_to_num(self, df, col : str) -> None:
        """Changes categories to binary columns

        Args:
            col (str): Column in DataFrame
            drop (bool, optional): Should it drop original column. Defaults to False.
        """
        categories = df[col].dropna().unique()
        features = []
        for cat in categories:
            binary = (df[col] == cat)
            df[cat] = binary.astype("int")
    
    def fillna(self, df, median : bool = False, mean : bool = False,
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
        
        for col in df:
            if df[col].dtype in [float, int]:
                if mean:
                    df[col].fillna(df[col].mean(), inplace = True)
                elif median:
                    df[col].fillna(df[col].median(), inplace = True)
                else:
                    df[col].fillna(-99999, inplace = True)
                    
            else:
                try:
                    if type(categories) == list:
                        for cat in categories:
                            self.cat_to_num(df, cat)
                    else:
                        self.cat_to_num(df, categories)
                 
                except KeyError:
                    raise KeyError(' "{}" is not a column in the DataFrame'.format(cat))

        if drop: df.drop(categories, axis=1, inplace=True)
        
        return gframe(df)
    
    def cfolds(self, df, y : str = 'target', n_splits : int = 5 ) -> tuple:
        """Creates validation and train split using folds

        Raises:
            KeyError: Column specified in target is not in DataFrame

        Returns:
            tuple (train_DataFrame, val_DataFrame): Train and Val split DataFrame
        """
        
        if y not in df:
            raise KeyError('{} is not in the DataFrame, did you forget to set y=column '.format(y))
        
        df['kfold'] = -1
        df = df.sample(frac=1).reset_index(drop=True)
        kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[y].values)):
            df.loc[val_idx, 'kfold'] = fold

        return (df.loc[df['kfold'] != 0], df.loc[df['kfold'] == 0])
        
    def train(self, models : list, y : Union[str, list] = 'target', x : Union[str, list] = 'Age') -> tuple:
        """Trains model using scikit learn models specified in models list

        Args:
            models (list): this list contains the various sckit learn models that are to be used to train
            y (Union[str, list], optional): Specifies the name on the column that is to be predicted. Defaults to 'target'.
            x (Union[str, list], optional): Specifies the name of the column that is USED to predict the target col. Defaults to 'Age'.

        Returns:
            tuple: [description]
        """
        
        for model in models:
            print("[MODEL] {}".format(model.__name__))
            for fold in range(2, 15):
                print(" ---> Fold: {}/15".format(fold))
                train, val = self.cfolds(self.df, y=y, n_splits=fold)
                
            print()
            
            
            
        print(train)
        
        # for model in models:
        #     model.fit(X=x, y=y)
        
        pass
        
# %%
from sklearn.linear_model import LinearRegression

# gframe(df).fillna(df, median=True, categories = ['Embarked'], drop = False)
# a.train(models=[LinearRegression], y='Survived', x='Age')
gframe(df).cfolds(df, y='Survived')[0]
# %%