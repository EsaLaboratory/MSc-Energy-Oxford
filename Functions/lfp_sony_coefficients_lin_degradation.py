from configparser import ConfigParser

import pandas as pd
import numpy as np

from coefficients_lin_degradation import CoefficientsLinearDegradation


class SonyLFPCoefficientsLinearDegradation(CoefficientsLinearDegradation):
    """
    MA Engwerth:    Class to get set points of the piecewise linear degradation model of the Sony LFP cell
    """

    def __init__(self):
        super().__init__()
        cal_degradation_file: str = '/Users/freddi/Documents/Frederik/MSc Energy Systems/Thesis/Bidding Strategy/Data/Linearzed Model/coefficients_linear_cal_deg_LFP.csv'
        self.__degradation_matrix = pd.read_csv(cal_degradation_file)
        self.__degradation_matrix = self.__degradation_matrix.values
        self.__len_degradation_matrix = self.__degradation_matrix.shape[1]
        self.__soh_vector_lower_limit = self.__degradation_matrix[1, 1:]

        cyc_degradation_file: str = '/Users/freddi/Documents/Frederik/MSc Energy Systems/Thesis/Bidding Strategy/Data/Linearzed Model/coefficients_linear_cycl_deg_LFP.csv'
        self.__degradation_matrix_cyc = pd.read_csv(cyc_degradation_file)
        self.__degradation_matrix_cyc = self.__degradation_matrix_cyc.values
        self.__len_degradation_matrix_cyc = self.__degradation_matrix_cyc.shape[1]
        self.__soh_vector_lower_limit_cyc = self.__degradation_matrix_cyc[1, 1:]
        self.__scaled_bool = 1
        self.__scaled_soh_base = 0.95
        self.__scaled_base_cal = self.get_cal_coefficients(self.__scaled_soh_base, first=True)[-1, 1]
        self.__scaled_base_cyc = self.get_cyc_coefficients(self.__scaled_soh_base, first=True)[-1, 1]

    def get_cal_coefficients(self, soh: float, first: bool = False) -> [float]:

        # Get the index of the set points fitting the actual SOH
        k = []
        if soh > self.__soh_vector_lower_limit[0]:
            k = 1
        elif soh <= self.__soh_vector_lower_limit[-2]:
            k = -1
        else:
            for v in range(0, int(self.__len_degradation_matrix-1)):
                if self.__soh_vector_lower_limit[v] >= soh > self.__soh_vector_lower_limit[v+1]:
                    k = v + 1
                    break
        # Get the corresponding values of calendar degradation for the set points of SOC
        coefficients = self.__degradation_matrix[2:13, k+1]

        coefficients_lin_cal_deg = np.zeros([11, 2])
        # Set points of SOC
        coefficients_lin_cal_deg[:, 0] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # Set points of calendar degradation
        coefficients_lin_cal_deg[:, 1] = coefficients

        if self.__scaled_bool and not first:
            base = coefficients_lin_cal_deg[-1, 1]
            scaling_factor = self.__scaled_base_cal/base
            coefficients_lin_cal_deg[:,1] = [x * scaling_factor for x in coefficients_lin_cal_deg[:,1]]

        return coefficients_lin_cal_deg

    def get_cyc_coefficients(self, soh: float, first: bool = False) -> [float]:

        # Get the index of the set points fitting the actual SOH
        n = []
        if soh > self.__soh_vector_lower_limit_cyc[0]:
            n = 1
        elif soh <= self.__soh_vector_lower_limit_cyc[-2]:
            n = -1
        else:
            for u in range(0, int(self.__len_degradation_matrix_cyc-1)):
                if self.__soh_vector_lower_limit_cyc[u] >= soh > self.__soh_vector_lower_limit_cyc[u+1]:
                    n = u + 1
                    break
        # Get the corresponding set points
        coefficients_cyc = self.__degradation_matrix_cyc[2:self.__degradation_matrix_cyc.shape[0], n+1]

        coefficients_lin_cyc_deg_cyc = np.zeros([28, 2])
        # Set points of energy throughput
        coefficients_lin_cyc_deg_cyc[:, 0] = coefficients_cyc[0:28]
        # Set points of cyclic degradation
        coefficients_lin_cyc_deg_cyc[:, 1] = coefficients_cyc[28:56]

        if self.__scaled_bool and not first:
            base = coefficients_lin_cyc_deg_cyc[-1, 1]
            scaling_factor = self.__scaled_base_cyc/base
            coefficients_lin_cyc_deg_cyc[:,1] = [x * scaling_factor for x in coefficients_lin_cyc_deg_cyc[:,1]]

        return coefficients_lin_cyc_deg_cyc
