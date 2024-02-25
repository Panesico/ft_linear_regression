# utils.py
# This file contains the utility functions for the project
import pandas as pd


def write_csv(theta0, theta1):
    thetas = {
        'theta0': [theta0],
        'theta1': [theta1]
    }
    data = pd.DataFrame(thetas)
    data.to_csv('thetas.csv', index=False)


def load_csv(filename):
    data = pd.read_csv(filename)
    return data


def normalize_data(mileages, prices):
    x = []
    y = []
    min_m = min(mileages)
    max_m = max(mileages)
    for mileage in mileages:
        x.append((mileage - min_m) / (max_m - min_m))
    min_p = min(prices)
    max_p = max(prices)
    for price in prices:
        y.append((price - min_p) / (max_p - min_p))
    return x, y


def normalize_value(column, elem):
    return (elem - min(column)) / (max(column) - min(column))


def denormalize_value(column, elem):
    return (elem * (max(column) - min(column))) + min(column)
