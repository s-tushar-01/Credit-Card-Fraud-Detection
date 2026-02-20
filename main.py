from src.preprocess import load_data, preprocess_data
from src.train import train_logistic, train_random_forest
from src.evaluate import evaluate_model

def main():
    df = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training Logistic Regression...")
    log_model = train_logistic(X_train, y_train)
    evaluate_model(log_model, X_test, y_test)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

if __name__ == "__main__":
    main()