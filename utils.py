import pandas as pd

def load_data(file):
    return pd.read_csv(file)

def get_insight(score):
    if score > 75:
        return "Excellent Performance 🎉"
    elif score > 60:
        return "Average Performance ⚠️"
    else:
        return "Needs Improvement ❗"