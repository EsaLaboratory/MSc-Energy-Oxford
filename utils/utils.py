import pandas as pd
import numpy as np

class Frequency:

    def __init__(self, frequency: float):
        self.__value = frequency
    
    def DM_high(self):
        if self.__value < 50.015:
            return 0
        elif self.__value < 50.1:
            return ((self.__value - 50.015) / 0.085) * 0.05
        elif self.__value < 50.2:
            return 0.05 + ((self.__value - 50.1) / 0.1) * 0.95
        else:
            return 1
        
    def DM_low(self):
        if self.__value > 49.985:
            return 0
        elif self.__value > 49.9:
            return ((-self.__value + 49.985) / 0.085) * 0.05
        elif self.__value > 49.8:
            return 0.05 + ((49.9 - self.__value) / 0.1) * 0.95
        else:
            return 1
        
    def DC_high(self):
        if self.__value < 50.015:
            return 0
        elif self.__value < 50.2:
            return ((self.__value - 50.015) / 0.185) * 0.05
        elif self.__value < 50.5:
            return 0.05 + ((self.__value - 50.2) / 0.3) * 0.95
        else:
            return 1
    
    def DC_low(self):
        if self.__value > 49.985:
            return 0
        elif self.__value > 49.8:
            return ((-self.__value + 49.985) / 0.185) * 0.05
        elif self.__value > 49.5:
            return 0.05 + ((49.8 - self.__value) / 0.3) * 0.95
        else:
            return 1
        
    def DR_high(self):
        if self.__value < 50.015:
            return 0
        else:
            return np.minimum(1, (self.__value - 50.015) / 0.1985)
    
    def DR_low(self):
        if self.__value > 49.985:
            return 0
        else:
            return np.minimum(1, (49.985 - self.__value) / 0.1985)

class FrequencyProfile(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def _constructor(self):
        return FrequencyProfile
    
    def add_response(self):
        self['DM_high'] = self['f'].apply(lambda x: Frequency(x).DM_high())
        self['DM_low'] = self['f'].apply(lambda x: Frequency(x).DM_low())
        self['DC_high'] = self['f'].apply(lambda x: Frequency(x).DC_high())
        self['DC_low'] = self['f'].apply(lambda x: Frequency(x).DC_low())
        self['DR_high'] = self['f'].apply(lambda x: Frequency(x).DR_high())
        self['DR_low'] = self['f'].apply(lambda x: Frequency(x).DR_low())

    def aggregate_to_EAC_windows(self):
        #convert the 'dtm' column to datetime
        self['dtm'] = pd.to_datetime(self['dtm'])

        # Resample the data to 4h intervals through aggregation of the mean
        self_resampled = self.resample('4h', on='dtm', offset='3h').mean()

        # Update the original DataFrame with the resampled data
        self.__dict__.update(self_resampled.__dict__)

        print('The data was resampled with a fixed offset of 3h. This means the first window starts on', self.index[0], 'and the last window starts on', self.index[-1])

    def add_EAC_window_identifier(self):
        self['EAC_window'] = (self.index.hour % 23 + 1) // 4

class AuctionData(pd.DataFrame):
    
    def __init__(self, data):
        super().__init__(data)

    def rearrange(self):
        self['deliveryStart'] = pd.to_datetime(self['deliveryStart'])
        # Create a pivot table
        pivot_df = self.pivot_table(
            index='deliveryStart',
            columns='auctionProduct',
            values='clearingPrice',
            aggfunc='last'  # Use 'last' to get the last occurrence if there are duplicates
        )
        # Reset the index to make 'deliveryStart' a column again
        pivot_df.reset_index(inplace=True)
        # Update the original DataFrame in place
        self.drop(self.index, inplace=True)
        self.columns = pivot_df.columns
        for col in pivot_df.columns:
            self[col] = pivot_df[col]
        self.index = pivot_df.index
        
        #rename the columns according to dict
        col_dict = {
            'deliveryStart': 'dtm',
            'DRH': 'DR_high',
            'DRL': 'DR_low',
            'DMH': 'DM_high',
            'DML': 'DM_low',
            'DCH': 'DC_high',
            'DCL': 'DC_low'
        }
        self.rename(columns=col_dict, inplace=True)
        self['dtm'] = pd.to_datetime(self['dtm'])
