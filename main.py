# Import required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


#  Load dataset
df = pd.read_csv("/content/drive/MyDrive/kc_house_data.csv")

#  Drop unnecessary columns
df1 = df.drop("id", axis=1)

#  Drop date column 
df2 = df1.drop("date", axis=1)


#  FEATURE ENGINEERING (Creating Proxy Variables)

# Income Proxy: average house price per zipcode
df2["income_proxy"] = df2.groupby("zipcode")["price"].transform("mean")

#  School Proxy: average house grade per zipcode
df2["school_proxy"] = df2.groupby("zipcode")["grade"].transform("mean")

#  Hospital Proxy: based on distance from city center

# Calculate center point (mean latitude & longitude)
center_lat = df2["lat"].mean()
center_long = df2["long"].mean()

# Calculate distance from center
df2["distance"] = ((df2["lat"] - center_lat)**2 + (df2["long"] - center_long)**2)**0.5

# Convert distance to hospital access score (closer = better)
df2["hospital_proxy"] = 1 / (df2["distance"] + 1)

#  Crime Proxy: inverse of income proxy (lower income → higher crime)
df2["crime_proxy"] = 1 / (df2["income_proxy"] + 1)


# NORMALIZATION (Scaling proxy values between 0–1)

df2["income_proxy"] = df2["income_proxy"] / df2["income_proxy"].max()
df2["school_proxy"] = df2["school_proxy"] / df2["school_proxy"].max()
df2["hospital_proxy"] = df2["hospital_proxy"] / df2["hospital_proxy"].max()
df2["crime_proxy"] = df2["crime_proxy"] / df2["crime_proxy"].max()


#  PREPARE DATA FOR MODEL
# Separate features (X) and target (y)
data = df2
X = data.drop("price", axis=1)   # Input features
y = data["price"]               # Target variable


#  FEATURE SCALING

# Scale features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#  TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# MODEL TRAINING (XGBoost)

# Create model (max_depth=100 )
model = XGBRegressor(max_depth=6)

# Train model
model.fit(X_train, y_train)


#  PREDICTION

# Predict on test data
y_pred = model.predict(X_test)


# MODEL EVALUATION

# Mean Squared Error
print("MSE:", mean_squared_error(y_test, y_pred))

# R2 Score (accuracy indicator)
print("R2 Score:", r2_score(y_test, y_pred))


#  MANUAL PREDICTION (Example)
model.predict([[11,34,567,343,13,64,2324,64,61,13,224,5546,24,256,64446,353,333,35335,7447,223543,3546,4647,3344]])
