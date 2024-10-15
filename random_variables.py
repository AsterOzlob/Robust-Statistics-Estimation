import math
from abc import ABC, abstractmethod

import numpy as np


class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        """
        Возвращает значение плотности вероятности (Probability Density Function - PDF)
        для заданного значения случайной величины.

        Параметры:
        x: float
            Значение случайной величины, для которого нужно вычислить плотность вероятности.

        Возвращает:
        float
            Значение плотности вероятности для заданного значения x.
        """
        pass

    @abstractmethod
    def cdf(self, x):
        """
        Возвращает значение функции распределения (Cumulative Distribution Function - CDF)
        для заданного значения случайной величины.

        Параметры:
        x: float
            Значение случайной величины, для которого нужно вычислить функцию распределения.

        Возвращает:
        float
            Значение функции распределения для заданного значения x.
        """
        pass

    @abstractmethod
    def quantile(self, alpha):
        """
        Возвращает квантиль уровня alpha для случайной величины.

        Параметры:
        alpha: float
            Уровень, для которого нужно вычислить квантиль. Должен быть в диапазоне [0, 1].

        Возвращает:
        float
            Квантиль уровня alpha для данной случайной величины.
        """
        pass


class UniformRandomVariable(RandomVariable):
    def __init__(self, left=0, right=1) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def pdf(self, x):
        if x < self.left or x > self.right:
            return 0
        else:
            return 1 / (self.right - self.left)

    def cdf(self, x):
        if x < self.left:
            return 0
        elif x > self.right:
            return 1
        else:
            return (x - self.left) / (self.right - self.left)

    def quantile(self, alpha):
        return self.left + alpha * (self.right - self.left)
