# Function to predict car price based on mileage using the trained parameters
import utils


def estimate_price(mileages, prices, thetas, mileage):
    return utils.denormalize_value(prices, thetas[0] + (thetas[1] * utils.normalize_value(mileages, mileage)))


# Main function
def main():
    # Prompt user for mileage input
    mileage = float(input("Enter the mileage of the car: "))
    mileages = utils.load_csv('data.csv').iloc[:, 0]
    prices = utils.load_csv('data.csv').iloc[:, 1]

    # Load the trained parameters theta0 and theta1
    # These values should have been saved after training the model
    # You should load the actual values of theta0 and theta1 from a file or database here
    thetas = utils.load_csv('thetas.csv').iloc[0]

    # Predict the price using the trained model
    estimated_price = estimate_price(mileages, prices, thetas, mileage)

    # Display the estimated price
    print("Estimated price for mileage {}: ${:.2f}".format(mileage, estimated_price))


# Entry point of the program
if __name__ == "__main__":
    main()
