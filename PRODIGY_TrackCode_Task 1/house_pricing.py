import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load the Kaggle data
# Make sure train.csv is in the same folder!
df = pd.read_csv('train.csv')

# 2. Select the columns required for the task
# Kaggle names: GrLivArea (SqFt), BedroomAbvGr (Beds), FullBath (Baths)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# 3. Clean the data (Kaggle data often has missing values)
# We filter for only the columns we need and drop any rows with empty cells
clean_df = df[features + [target]].dropna()

X = clean_df[features]
y = clean_df[target]

# 4. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Test it with a custom house
# Let's predict a 2000 sqft house with 3 beds and 2 baths
custom_house = [[2000, 3, 2]] 
prediction = model.predict(custom_house)

print(f"Predicted Price for Kaggle House: ${prediction[0]:,.2f}")