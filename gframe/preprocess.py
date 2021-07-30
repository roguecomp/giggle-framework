from typing import Union
import pandas as pd
import gframe

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
    
def fillna(self, median : bool = False, mean : bool = False, func = None,
           categories : Union[str, list] = [], drop : bool = False ) -> pd.DataFrame:
    """fills null values in data frame with functions either custom or builtin.

    Args:
    -----
        median (bool, optional):
            fill null values with median. Defaults to False.
            
        mean (bool, optional):
            fill null values with mean. Defaults to False.
            
        func (function, optional):
            fill null values in DataFrame with custom function.
            func takes parameters func(self.df, col).
            
        categories (Union[str, list], optional):
            changes columns to boolean columns. Defaults to [].
            
        drop (bool, optional):
            drops original column after making them boolean. Defaults to False.

    Raises:
        KeyError: If passed in column from list is not in DataFrame exception in raised
    """
    
    for col in self.df:
        if self.df[col].dtype in [float, int]:
            if mean:
                self.df[col].fillna(self.df[col].mean(), inplace = True)
            elif median:
                self.df[col].fillna(self.df[col].median(), inplace = True)
            elif func != None:
                func(self.df, col)
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
    
    return gframe.gframe(self.df)
