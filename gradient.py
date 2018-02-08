import matplotlib.pyplot as plt
import numpy as np


def compute_error(points, b, m):
    X = points[:, 0]
    y = points[:, 1]
    h = (X * m) + b

    return sum(y - h)


def gradient_step(points, b, m, learning_rate):
    X = points[:, 0]
    y = points[:, 1]
    N = float(len(points))
    h = (X * m) + b

    b_gradient = -(2/N) * sum(y - h)
    m_gradient = -(2/N) * sum(X * (y - h))

    b -= learning_rate * b_gradient
    m -= learning_rate * m_gradient

    return b, m


def gradient_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = gradient_step(points, b, m, learning_rate)

        print("Error after %d iterations: %.6f" % (i+1, compute_error(points, b, m)))

    return [b, m]


def run():
    points = np.genfromtxt("data.csv", delimiter=',')
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    [b, m] = gradient_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    X = points[:, 0]
    y = points[:, 1]
    calc_points = X * m + b

    plt.title('Basic linear regression with gradient descent')
    plt.scatter(X, y)
    plt.plot(X, calc_points)
    plt.show()


if __name__ == '__main__':
    run()