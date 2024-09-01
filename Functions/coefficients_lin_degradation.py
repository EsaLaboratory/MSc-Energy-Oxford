from abc import ABC, abstractmethod

from simses.commons.state.system import SystemState


class CoefficientsLinearDegradation(ABC):
    """
    MA Engwerth:    Abstract class for piecewise linear degradation models, which can be used in optimized operation
                    strategies
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_cal_coefficients(self, soh: float) -> [float]:
        """
        Returns the set points of the piecewise linear calendar degradation model

        Returns:
        -------
        float
        """
        pass

    @abstractmethod
    def get_cyc_coefficients(self, soh: float) -> [float]:
        """
        Returns the set points of the piecewise linear cyclic degradation model

        Returns:
        -------
        float
        """
        pass
