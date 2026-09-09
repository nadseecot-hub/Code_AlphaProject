import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("car_data.csv")
print("Shape:", df.shape)
print(df.head())
print(df.info())


CURRENT_YEAR = 2024
df['Car_Age'] = CURRENT_YEAR - df['Year']
df.drop('Year', axis=1, inplace=True)


df.drop('Car_Name', axis=1, inplace=True)


df_encoded = pd.get_dummies(
    df,
    columns=['Fuel_Type', 'Selling_type', 'Transmission'],
    drop_first=True   # avoid dummy variable trap (multicollinearity)
)

print("\nColumns after encoding:", df_encoded.columns.tolist())

# -----------------------------
# 3. EDA visuals (saved as images for your README)
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x='Present_Price', y='Selling_Price')
plt.title("Present Price vs Selling Price")
plt.tight_layout()
plt.savefig("price_scatter.png")
plt.close()

# -----------------------------
# 4. Train / Test Split
# -----------------------------
X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model 1: Linear Regression (baseline)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_preds = lin_model.predict(X_test)


# Model 2: Random Forest Regressor

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# Evaluation

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n {name} ")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

results = []
results.append(evaluate("Linear Regression", y_test, lin_preds))
results.append(evaluate("Random Forest", y_test, rf_preds))

results_df = pd.DataFrame(results)
print("\nComparison Table:\n", results_df)
results_df.to_csv("model_comparison.csv", index=False)


importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):\n", importances)

plt.figure(figsize=(7, 5))
importances.plot(kind='bar')
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# best model: Random Forest, plot actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, rf_preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

