import numpy as np
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd

btc = yf.download("BTC-USD", start="2015-01-01", end="2024-07-30")
btc.dropna(inplace=True)

gold = yf.download("GC=F", start="2015-01-01", end="2024-07-30")
gold.dropna(inplace=True)

btc["Gold_Close"] = gold["Close"]

btc["Tomorrow"] = btc["Close"].shift(-1)
btc.dropna(inplace=True)
y = btc["Tomorrow"]
X = btc[["Open", "Close", "Low", "High", "Gold_Close"]]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(xTrain, yTrain)

allPred = model.predict(X)
yTrainPred = model.predict(xTrain)
yTestPred = model.predict(xTest)

allRmse = np.sqrt(mean_squared_error(y, allPred))
TrainRmse = np.sqrt(mean_squared_error(yTrain, yTrainPred))
TestRmse = np.sqrt(mean_squared_error(yTest, yTestPred))
print("All RMSE:", allRmse)
print("Train RMSE:", TrainRmse)
print("Test RMSE:", TestRmse)

last_known_values = btc.iloc[-1][["Open", "Close", "Low", "High", "Gold_Close"]].to_frame().T
future_predictions = []

for _ in range(100):  # پیش‌بینی برای ۱۰ روز آینده
    next_prediction = model.predict(last_known_values)
    future_predictions.append(next_prediction[0])
    last_known_values = pd.DataFrame([[
        last_known_values["Open"].values[0],
        next_prediction[0],
        last_known_values["Low"].values[0],
        last_known_values["High"].values[0],
        last_known_values["Gold_Close"].values[0]
    ]], columns=["Open", "Close", "Low", "High", "Gold_Close"])

future_dates = pd.date_range(start=btc.index[-1] + pd.Timedelta(days=1), periods=100, freq='D')
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=["Predicted Close"])

plt.figure(figsize=(12, 6))
plt.plot(btc.index, y, color='red', label='BTC Close')
plt.plot(btc.index, allPred, color='black', linewidth=0.5, label='Predicted Tomorrow')
#plt.plot(xTrain.index, yTrainPred, color='blue', label='Train')
plt.plot(xTest.index, yTestPred, color='orange', label='Test')
#plt.plot(future_df.index, future_df["Predicted Close"], color='green', label='Future Prediction (10 days)')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted closing prices for the next 10 days:\n{future_df}")
