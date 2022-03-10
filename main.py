import math
import numpy as np
import matplotlib.pyplot as plt

def function(t: float) -> float:
    return math.exp(t - 1)

def train(learning_rate, p: int, epochs: int, a = -2.0, b = 2, m = 20):
    step = (b - a) / m
    t = np.arange(a, b, step)
    weights = np.zeros(p + 1, dtype=float)
    x = np.array([function(time) for time in t])

    if len(t) < p:
        raise Exception('Size of window is greater than amount of points')

    for epoch in range(0, epochs):
        for i in range(0, m - p):
            input_signals = np.concatenate(([1.0,], np.array([x[i + j] for j in range(0, p)])))
            output = weights.dot(input_signals)
            error = x[i + p] - output
            weights += input_signals * error * learning_rate

    x_calculated = np.empty(m - p)
    x_real = x[p:]
    for i in range(0, m - p):
        input_signals = np.concatenate(([1.0,], np.array([x[i + j] for j in range(0, p)])))
        x_calculated[i] = weights.dot(input_signals)

    return x_calculated, np.std(x_real - x_calculated), weights

def approximate(x, w, p, m, a = 2, b = 6):
    for i in range(0, m):
        input_signals = np.concatenate(([1.0,], np.array(x[len(x) - p:])))
        new_point = w.dot(input_signals)
        x = np.append(x, new_point)
    return x[m - p:]

def get_standard_deviation(x_approximated, a = -2.0, b = 2.0, m = 20):
    step = ((2 * b - a) - b) / 20
    time = np.arange(b, 2 * b - a, step)
    x_real = [function(t) for t in time]

    return np.std(x_approximated - x_real)

def draw_plot(x, precision = 100, a = -2.0, b = 2.0):
    short_step = (2 * b - a - a) / precision
    plot_data_t = np.arange(a, 2*b-a, short_step)
    plot_data_x = np.array([function(time) for time in plot_data_t])
    plt.plot(plot_data_t, plot_data_x, color='tab:blue')

    new_a = b
    new_b = 2 * b - a
    short_step = (new_b - b) / 20

    plt.scatter(np.arange(new_a, new_b, short_step), x, color='tab:orange', linewidths=0.00001)

    plt.show()

#FIRST TASK
m = 20
p = 4
eta = 1

plot_deviations = list()
plot_epochs = range(5, 500, 5)

for M in plot_epochs:
    x_trained, standard_derivation, weights = train(eta, p, M)
    x_approximated = approximate(x_trained, weights, p, m)
    plot_deviations.append(get_standard_deviation(x_approximated))

plt.plot(plot_epochs[1:], plot_deviations[1:])
plt.show()

#SECOND TASK
M = 200
m = 20
p = 4
eta = 1

plot_deviations = list()
plotEta = np.arange(0.05, 1.2, 0.01)

for eta in plotEta:
    x_trained, standard_derivation, weights = train(eta, p, M)
    x_approximated = approximate(x_trained, weights, p, m)
    plot_deviations.append(get_standard_deviation(x_approximated))

plt.plot(plotEta[1:], plot_deviations[1:])
plt.show()

#THIRD TASK
m = 20
M = 500
p = 6
eta = 0.5

x_trained, standard_derivation, weights = train(eta, p, M)
x_approximated = approximate(x_trained, weights, p, m)
print('Weights:', weights)
print('Standard deviation =', get_standard_deviation(x_approximated))
draw_plot(x_approximated)