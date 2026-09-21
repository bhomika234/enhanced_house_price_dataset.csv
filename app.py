import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="House Price Prediction & Model Comparison",
    page_icon="🏠",
    layout="wide",
)


# --- 1. Load and Cache Dataset & Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
  # Dataset load karein (Ensure karein ki enhanced_house_price_dataset.csv yahin maujood ho)
  df = pd.read_csv("enhanced_house_price_dataset.csv")

  # Handle missing values (jaise notebook mein kiya tha)
  num_cols = df.select_dtypes(include=np.number).columns
  cat_cols = df.select_dtypes(exclude=np.number).columns

  for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

  for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

  # Remove duplicates
  df = df.drop_duplicates().reset_index(drop=True)
  return df


df = load_and_preprocess_data()

# --- Main App Title ---
st.title("🏠 House Price Prediction & ML Model Comparison Dashboard")
st.markdown(
    "Task 02 – Build and Compare Machine Learning Models using Regression"
)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset Overview & EDA",
        "Train & Compare Models",
        "Predict House Price",
    ],
)

# Features and Target split
X = df.drop(columns=["Price"])
y = df["Price"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


# --- SECTION 1: Dataset Overview & EDA ---
if app_mode == "Dataset Overview & EDA":
  st.header("📊 Dataset Overview")
  col1, col2, col3 = st.columns(3)
  col1.metric("Total Records", df.shape[0])
  col2.metric("Total Features", df.shape[1] - 1)
  col3.metric("Target Variable", "Price")

  st.subheader("Sample Data (First 5 Rows)")
  st.dataframe(df.head())

  st.subheader("Dataset Summary Statistics")
  st.dataframe(df.describe())

  st.subheader("Missing Values Check")
  missing_df = pd.DataFrame(
      {"Missing Values": df.isnull().sum(), "Data Type": df.dtypes}
  )
  st.dataframe(missing_df)


# --- SECTION 2: Train & Compare Models ---
elif app_mode == "Train & Compare Models":
  st.header("⚙️ Model Training & Evaluation Performance")

  if st.button("Run & Train All Models"):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "K-Neighbors": KNeighborsRegressor(),
        "SVR": SVR(),
    }

    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models.items()):
      status_text.text(f"Training {name}...")

      pipeline = Pipeline(
          steps=[("preprocessor", preprocessor), ("regressor", model)]
      )

      start_time = time.time()
      pipeline.fit(X_train, y_train)
      train_time = time.time() - start_time

      y_pred = pipeline.predict(X_test)

      mae = mean_absolute_error(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      rmse = np.sqrt(mse)
      r2 = r2_score(y_test, y_pred)

      results.append(
          {
              "Model": name,
              "R2 Score": round(r2, 4),
              "MAE": round(mae, 2),
              "RMSE": round(rmse, 2),
              "Training Time (s)": round(train_time, 4),
          }
      )
      progress_bar.progress((idx + 1) / len(models))

    status_text.text("Training Complete!")
    results_df = pd.DataFrame(results).sort_values(
        by="R2 Score", ascending=False
    )

    st.subheader("🏆 Model Comparison Leaderboard")
    st.dataframe(results_df, use_container_width=True)

    st.success(
        f"Best performing model based on R2 Score is:"
        f" **{results_df.iloc[0]['Model']}**"
    )
  else:
    st.info("Click the button above to train models and compare their metrics.")


# --- SECTION 3: Predict House Price ---
elif app_mode == "Predict House Price":
  st.header("🔮 Predict House Price")
  st.write(
      "Provide the property details below to estimate the predicted house"
      " price."
  )

  # Input form layout in sidebar or main area columns
  col1, col2, col3 = st.columns(3)

  with col1:
    area = st.number_input("Area (sq ft)", min_value=400, max_value=6000, value=3000)
    bedrooms = st.selectbox("Bedrooms", sorted(df["Bedrooms"].unique()))
    bathrooms = st.selectbox("Bathrooms", sorted(df["Bathrooms"].unique()))
    stories = st.selectbox("Stories", sorted(df["Stories"].unique()))

  with col2:
    parking = st.selectbox("Parking Spaces", sorted(df["Parking"].unique()))
    age = st.number_input("Age of House (Years)", min_value=0, max_value=50, value=15)
    city = st.selectbox("City", df["City"].unique())
    furnishing = st.selectbox("Furnishing", df["Furnishing"].unique())

  with col3:
    main_road = st.selectbox("Main Road Access", df["Main Road"].unique())
    guest_room = st.selectbox("Guest Room", df["Guest Room"].unique())
    basement = st.selectbox("Basement", df["Basement"].unique())
    water_supply = st.selectbox("Water Supply", df["Water Supply"].unique())

  col4, col5 = st.columns(2)
  with col4:
    air_conditioning = st.selectbox(
        "Air Conditioning", df["Air Conditioning"].unique()
    )
    preferred_tenant = st.selectbox(
        "Preferred Tenant", df["Preferred Tenant"].unique()
    )
  with col5:
    locality_rating = st.slider("Locality Rating", 1, 10, 5)

  # Collect user input into DataFrame
  input_data = pd.DataFrame({
      "Area": [area],
      "Bedrooms": [bedrooms],
      "Bathrooms": [bathrooms],
      "Stories": [stories],
      "Parking": [parking],
      "Age": [age],
      "City": [city],
      "Furnishing": [furnishing],
      "Main Road": [main_road],
      "Guest Room": [guest_room],
      "Basement": [basement],
      "Water Supply": [water_supply],
      "Air Conditioning": [air_conditioning],
      "Preferred Tenant": [preferred_tenant],
      "Locality Rating": [locality_rating],
  })

  if st.button("Predict Price", type="primary"):
    # Train Random Forest by default for prediction pipeline
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )
    model_pipeline.fit(X_train, y_train)

    prediction = model_pipeline.predict(input_data)[0]

    st.success(f"### Estimated House Price: ₹ {prediction:,.2f}")