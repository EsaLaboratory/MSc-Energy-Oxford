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
            return ((-self.value + 49.985) / 0.085) * 0.05
        elif self.__value > 49.8:
            return 0.05 + ((49.9 - self.value) / 0.1) * 0.95
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
            return np.minimum(1, (50.015 - self.__value) / 0.1985)