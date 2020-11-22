import requests
import pandas as pd
import scipy
from scipy import stats
# from sklearn.linear_model import LinearRegression as LinRegr
# from sklearn.metrics import mean_squared_error
import numpy 
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE

    train_df = pd.read_csv(TRAIN_DATA_URL, header=None, index_col=0).to_numpy()
    test_df = pd.read_csv(TEST_DATA_URL, header=None, index_col=0).to_numpy()


    x_train = train_df[0,:]
    y_train = train_df[1,:]
    x_test = test_df[0,:]
    y_test = test_df[1,:]


    # linr = LinRegr().fit(x_train.reshape(-1,1), y_train)


    # rmse_test = mean_squared_error(y_test, linr.predict(x_test.reshape(-1,1)), squared=False)
    # print(rmse_test)

    # return linr.predict(area.reshape(-1,1))

    slope, intercept, _,_,_ = stats.linregress(x_train, y_train)
    # print(f"slope is {slope}, intercept is {intercept}")

    preds = (slope*area)+intercept

    # print(area.shape, preds.shape)

    return preds






if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
