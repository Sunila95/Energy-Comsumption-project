import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from energy_model import preprocess_data, train_models, evaluate_models, save_best_model, visualize_results

# Load dataset
df = pd.read_csv("energydata_complete.csv")

# Preprocess
X, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = train_models()

# Evaluate
results, best_model_name, best_model = evaluate_models(models, X_train, X_test, y_train, y_test)

# Save best model
save_best_model(best_model)

# Visualize results
visualize_results(best_model, best_model_name, X_test, y_test, df)
