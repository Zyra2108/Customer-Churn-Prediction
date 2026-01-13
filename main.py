import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
REQUIRED_COLUMNS = {"monthly_usage", "tenure", "churn"}
def preprocess(df):
    df = df.dropna()
    return df

def engineer_features(df):
    df["usage_ratio"] = df["monthly_usage"] / (df["tenure"] + 1)
    return df
  
def run_pipeline(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        print("Failed to read CSV file.")
        return

    if not REQUIRED_COLUMNS.issubset(df.columns):
        print("Dataset missing required columns:", REQUIRED_COLUMNS)
        return

    df = preprocess(df)
    df = engineer_features(df)
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n--- Model Evaluation ---\n")
    print(classification_report(y_test, preds))
def main():
    path = input("D:\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    run_pipeline(path)
if __name__ == "__main__":
    main()
