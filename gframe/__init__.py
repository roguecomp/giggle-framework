# %%
import pandas as pd
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
    
    # Preprocessing tools for machine learning 
    from gframe.preprocess import cat_to_num, fillna

    # Training tools for machine learning
    from gframe.train import cfolds, train 
