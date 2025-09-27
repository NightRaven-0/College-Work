#23BAI11010 LR
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset (replace with your path)
train_X = pd.read_csv("R:\VS CODE\The Apps\CP\X_test_scaled_linear.csv")
train_y = pd.read_csv("R:\VS CODE\The Apps\CP\y_train.csv")
test_X = pd.read_csv("R:\VS CODE\The Apps\CP\X_train_scaled_linear.csv")
test_y = pd.read_csv("R:\VS CODE\The Apps\CP\y_test.csv")

# Initialize and fit model
model = LinearRegression()
model.fit(train_X, train_y)

# Predictions
y_pred = model.predict(test_X)

# Results
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", model.score(test_X, test_y))