from abc import abstractmethod, ABC
import random

import numpy as np

from random_variables import RandomVariable


# Напишем интерфейс для генераторов случайных величин.
class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass


# Реализуем конкретный класс генератора случайных величин на базе квантилей.
class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)


class TukeyNumberGenerator(RandomNumberGenerator):
    def __init__(self, basis: RandomVariable, emissions: RandomVariable):
        super().__init__(basis)
        self.basis = basis
        self.emissions = emissions

    def get(self, N: int, epsilon: float):
        """Генерирует N чисел, используя симметричный метод Тьюки."""
        sample = []
        for _ in range(N):
            if random.random() < epsilon:
                # Выбираем значения из emissions с вероятностью 10%
                sample.append(self.emissions.quantile(random.random()))
            else:
                # Иначе выбираем значения из basis с вероятностью 90%
                sample.append(self.basis.quantile(random.random()))
        return sample


class AsymmetricTukeyNumberGenerator:
    def __init__(self, basis: RandomVariable, emissions: RandomVariable):
        self.basis = basis
        self.emissions = emissions

    def get(self, N: int, epsilon: float):
        us = np.random.uniform(0, 1, N)
        sample = []
        for i in us:
            if i < epsilon:
                sample.append(self.emissions.quantile(random.random()))
            else:
                sample.append(self.basis.quantile(random.random()))
        return sample
