import matplotlib.pyplot as plt
import numpy as np

from estimations import TruncatedMean
from modelling import bootstrap_resample, Modelling
from random_number_generator import SimpleRandomNumberGenerator, TukeyNumberGenerator, AsymmetricTukeyNumberGenerator
from random_variables import UniformRandomVariable

POINTS = 100


def print_statistics(alpha_values, bias_sqr_values, var_values, mse_values):
    best_alpha_sqr = alpha_values[np.argmin(bias_sqr_values)]
    best_alpha_var = alpha_values[np.argmin(var_values)]
    best_alpha_mse = alpha_values[np.argmin(mse_values)]

    print(f'\nНаилучшее значение alpha при квадратах смещения: {best_alpha_sqr}')
    print(f'Наилучшее значение alpha при дисперсии: {best_alpha_var}')
    print(f'Наилучшее значение alpha при СКО: {best_alpha_mse}')


def plot_samples(alpha_values, bias_sqr_values, var_values, mse_values):
    plt.figure(figsize=(10, 6))

    display_ticks = np.arange(0, 0.501, 0.05)

    # Plot for bias squared values
    plt.subplot(3, 1, 1)
    plt.plot(alpha_values, bias_sqr_values, label='Квадрат смещения')
    plt.xlabel('Альфа')
    plt.ylabel('Квадрат смещения')
    plt.title('Квадрат смещения & Альфа')
    plt.grid(True)
    plt.legend()
    plt.xticks(display_ticks, rotation=45)

    # Plot for variance values
    plt.subplot(3, 1, 2)
    plt.plot(alpha_values, var_values, label='Дисперсия')
    plt.xlabel('Альфа')
    plt.ylabel('Дисперсия')
    plt.title('Дисперсия & Альфа')
    plt.grid(True)
    plt.legend()
    plt.xticks(display_ticks, rotation=45)

    # Plot for MSE values
    plt.subplot(3, 1, 3)
    plt.plot(alpha_values, mse_values, label='СКО')
    plt.xlabel('Альфа')
    plt.ylabel('СКО')
    plt.title('СКО & Альфа')
    plt.grid(True)
    plt.legend()
    plt.xticks(display_ticks, rotation=45)

    plt.tight_layout()
    plt.show()


def run_modelling(number_resample: int, location: float, resamples: list, N):
    # Поиск наилучшего alpha
    alpha_values = [i for i in np.arange(0.001, 0.501, 0.001)]
    mse_values = []  # СКО
    var_values = []  # Дисперсия
    bias_sqr_values = []  # Квадрат смещения

    for alpha in alpha_values:
        tm_estimator = TruncatedMean(alpha)
        tm_sample = []
        for i in range(number_resample):
            sample = resamples[i]
            tm = tm_estimator.estimate(sample)
            tm_sample.append(tm)

        modelling = Modelling(tm_sample, location, number_resample)
        modelling.run()

        bias_sqr_values.append(modelling.estimate_bias_sqr())
        var_values.append(modelling.estimate_var())
        mse_values.append(modelling.estimate_mse())

    return alpha_values, mse_values, var_values, bias_sqr_values


def without_emissions(location: float, scale: float, N: int, number_resample: int):
    rv = UniformRandomVariable(location, scale)
    generator = SimpleRandomNumberGenerator(rv)

    sample = generator.get(N)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, location, resamples, N)

    alpha_values = modelling[0]
    mse_values = modelling[1]
    var_values = modelling[2]
    bias_sqr_values = modelling[3]

    print_statistics(alpha_values, bias_sqr_values, var_values, mse_values)

    plot_samples(alpha_values, bias_sqr_values, var_values, mse_values)


def with_symmetrical_emissions(location: float, scale: float, N: int, number_resample: int):
    rv = UniformRandomVariable(location, scale)
    tukey_rv = UniformRandomVariable(-4, 5)

    generator = TukeyNumberGenerator(rv, tukey_rv)

    sample = generator.get(N, 0.1)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, (location + scale) / 2, resamples, N)

    alpha_values = modelling[0]
    mse_values = modelling[1]
    var_values = modelling[2]
    bias_sqr_values = modelling[3]

    print_statistics(alpha_values, bias_sqr_values, var_values, mse_values)

    plot_samples(alpha_values, bias_sqr_values, var_values, mse_values)


def with_asymmetrical_emissions(location: float, scale: float, N: int, number_resample: int):
    rv = UniformRandomVariable(location, scale)

    tukey_rv = UniformRandomVariable(7, 10)

    generator = AsymmetricTukeyNumberGenerator(rv, tukey_rv)

    sample = generator.get(N, 0.1)

    resamples = bootstrap_resample(sample, number_resample)

    modelling = run_modelling(number_resample, (location + scale) / 2, resamples, N)

    alpha_values = modelling[0]
    mse_values = modelling[1]
    var_values = modelling[2]
    bias_sqr_values = modelling[3]

    print_statistics(alpha_values, bias_sqr_values, var_values, mse_values)

    plot_samples(alpha_values, bias_sqr_values, var_values, mse_values)
