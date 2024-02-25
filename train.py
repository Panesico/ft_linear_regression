# train
import os
import utils
import prediction
import matplotlib.pyplot as plt

# Load the data

ITERATIONS = int(os.getenv('ITERATIONS', 1500))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.1))


def train(file):
    # Normalize the mileage and price columns
    normalized_mileage, normalized_price = utils.normalize_data(file.iloc[:, 0], file.iloc[:, 1])

    t0_history = [0.0]
    t1_history = [0.0]

    theta0 = 0
    theta1 = 0
    m = file.shape[0]
    for _ in range(ITERATIONS):
        gradient_theta0 = 0
        gradient_theta1 = 0
        for i in range(m):
            gradient_theta0 += ((theta0 + normalized_mileage[i] * theta1) - normalized_price[i])
            gradient_theta1 += ((theta0 + normalized_mileage[i] * theta1) - normalized_price[
                i]) * normalized_mileage[i]

        t0_history.append(theta0)
        t1_history.append(theta1)

        theta0 -= LEARNING_RATE * gradient_theta0 / m
        theta1 -= LEARNING_RATE * gradient_theta1 / m

    return theta0, theta1, t0_history, t1_history


def main():
    file = utils.load_csv('data.csv')
    mileages = file.iloc[:, 0]
    prices = file.iloc[:, 1]
    theta0, theta1, t0_history, t1_history = train(file)
    lineX = [float(min(mileages)), float(max(mileages))]
    lineY = []
    for value in lineX:
        value = theta1 * utils.normalize_value(mileages, value) + theta0
        lineY.append(utils.denormalize_value(prices, value))
    plt.plot(mileages, prices, 'bo', lineX, lineY, 'r-')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()
    utils.write_csv(theta0, theta1)


if __name__ == '__main__':
    main()
