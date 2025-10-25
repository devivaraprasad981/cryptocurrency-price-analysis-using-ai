import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error




def train_and_save_model(X, y, path='models/rf_model.pkl'):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print('MSE on holdout:', mse)
joblib.dump(model, path)
return path




def load_model_predict(df, model_path='models/rf_model.pkl'):
# build features from the latest df and predict next n points (simple rolling)
X, y = None, None
from data_utils import make_features
X, y = make_features(df)
model = joblib.load(model_path)
preds = model.predict(X)
# return last 10 predictions aligned with timestamps
return list(preds[-10:])