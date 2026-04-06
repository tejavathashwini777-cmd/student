import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models(data):
    X = data[["StudyHours", "Attendance", "PreviousMarks"]]
    y = data["FinalMarks"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr = LinearRegression()
    rf = RandomForestRegressor()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    return lr, rf, X_test, y_test