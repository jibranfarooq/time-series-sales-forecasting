import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Sales Forecasting Pro",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071329, #17143b);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stAlert"] p {
    color: #1f2937 !important;
}

div[data-testid="stAlert"] {
    color: #1f2937 !important;
}
div[data-testid="stMetric"] {
    background: #20264d;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #7257d9;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 SALES FORECASTING PRO")
st.caption("Time Series Analysis | Machine Learning Dashboard")

st.sidebar.title("⚙️ Dashboard Menu")
page = st.sidebar.radio(
    "Choose Section",
    ["Dashboard", "Sales Analysis", "Future Prediction"]
)

months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

sales = [12000, 15000, 13000, 18000, 20000, 25000,
         22000, 27000, 30000, 28000, 35000, 40000]

X = [[i] for i in range(1, 13)]
model = LinearRegression()
model.fit(X, sales)

if page == "Dashboard":
    st.header("Business Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"₹{sum(sales):,.0f}")
    c2.metric("Average Sales", f"₹{sum(sales)/len(sales):,.2f}")
    c3.metric("Highest Sales", f"₹{max(sales):,.0f}")

    st.subheader("Monthly Sales Performance")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(months, sales, marker="o", linewidth=3)
    ax.set_ylabel("Sales (₹)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.2)
    fig.patch.set_facecolor("#17143b")
    ax.set_facecolor("#20264d")
    st.pyplot(fig)

elif page == "Sales Analysis":
    st.header("📈 Sales Analysis")

    selected_month = st.selectbox("Select Month", months)
    index = months.index(selected_month)

    st.metric("Sales in " + selected_month, f"₹{sales[index]:,}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(months, sales)
    ax.set_ylabel("Sales (₹)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

elif page == "Future Prediction":
    st.header("🔮 Machine Learning Prediction")

st.success("📂 Upload your sales CSV file below to generate predictions from your own data.")    

st.warning("ℹ️ Predictions are based on the sample sales dataset.")

uploaded_file = st.file_uploader(
    "Upload your sales CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    st.success("CSV File Uploaded Successfully!")

    st.subheader("Uploaded Sales Data")
    st.dataframe(uploaded_data)

    st.subheader("📊 Monthly Sales Graph")
    st.line_chart(uploaded_data.set_index("Month")["Sales"])

    st.subheader("🏆 Highest and Lowest Sales")
    col1, col2 = st.columns(2)

    col1.metric(
        "Highest Sales",
        f"₹{uploaded_data['Sales'].max():,.0f}"
    )

    col2.metric(
        "Lowest Sales",
        f"₹{uploaded_data['Sales'].min():,.0f}"
    )


    st.subheader("Sales Summary")
    st.write("Total Sales:", uploaded_data["Sales"].sum())
    st.write("Average Sales:", uploaded_data["Sales"].mean())
    st.divider()
    st.subheader("🔮 Prediction from Uploaded Data")

    uploaded_X = [[i + 1] for i in range(len(uploaded_data))]
    uploaded_y = uploaded_data["Sales"].tolist()
    split_index = int(len(uploaded_X) * 0.8)

    X_train = uploaded_X[:split_index]
    X_test = uploaded_X[split_index:]

    y_train = uploaded_y[:split_index]
    y_test = uploaded_y[split_index:]
    test_model = LinearRegression()
    test_model.fit(X_train, y_train)

    test_predictions = test_model.predict(X_test)

    accuracy = r2_score(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = mean_squared_error(y_test, test_predictions) ** 0.5

    st.metric("Model R² Score", f"{accuracy:.2%}")
    st.metric("Mean Absolute Error", f"₹{mae:,.0f}")
    st.metric("Root Mean Squared Error", f"₹{rmse:,.0f}")
    st.subheader("📊 Actual vs Predicted Sales")

    plt.figure(figsize=(10, 5))

    plt.plot(
            range(len(y_test)),
            y_test,
            marker="o",
            label="Actual Sales"
        )

    plt.plot(
            range(len(test_predictions)),
            test_predictions,
            marker="o",
            label="Predicted Sales"
        )

    plt.xlabel("Test Data Months")
    plt.ylabel("Sales (₹)")
    plt.title("Actual vs Predicted Sales Comparison")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)
    plt.close()
    uploaded_model = LinearRegression()
    uploaded_model.fit(uploaded_X, uploaded_y)
        
    future_months = [
        [len(uploaded_data) + 1],
        [len(uploaded_data) + 2],
        [len(uploaded_data) + 3]
    ]

    future_predictions = uploaded_model.predict(future_months)

    st.subheader("📅 Next 3 Months Sales Forecast")

    col1, col2, col3 = st.columns(3)

    col1.metric(
       "January",
        f"₹{future_predictions[0]:,.0f}"
    )

    col2.metric(
       "February",
        f"₹{future_predictions[1]:,.0f}"
    )

    col3.metric(
        "March",
        f"₹{future_predictions[2]:,.0f}"
    )