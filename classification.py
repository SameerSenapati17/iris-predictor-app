import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import streamlit.components.v1 as components

# --- Custom CSS for better look ---
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .stButton>button { color: white; background: #4CAF50; }
    .stSidebar { background-color: #e3e6f0; color: #222 !important; }
    section[data-testid="stSidebar"] * { color: #222 !important; }
    /* Force selectbox (dropdown) in sidebar to be light */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #f9fafb !important;
        color: #222 !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        color: #222 !important;
        background-color: #f9fafb !important;
    }
    /* SHAP Explanation Header Styling */
    .shap-section {
        background: #2a2d34;
        border-radius: 16px;
        padding: 1.5em;
        margin: 2em 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        color: #e0e0e0;
        transition: all 0.3s ease-in-out;
    }

    .shap-section h4 {
        color: #81c784 !important;
        font-size: 1.6em;
        font-weight: 700;
        margin-bottom: 1em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }

    .shap-section h4::before {
        content: "🔍";
        font-size: 1.2em;
    }

    /* White background for the embedded SHAP JS force plot */
    .shap-section .js-plotly-plot,
    .shap-section .shap-plot,
    .shap-section > div > div {
        background: #ffffff !important;
        border-radius: 12px;
        border: 1px solid #ddd !important;
        padding: 1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Style the file uploader background and text */
    section[data-testid="stSidebar"] .stFileUploader {
        background-color: #f9fafb !important;
        border: 2px dashed #ccc !important;
        border-radius: 10px !important;
        padding: 1em !important;
        color: #333 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Style the browse button */
    section[data-testid="stSidebar"] .stFileUploader button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }

    /* Style the drag-and-drop text */
    section[data-testid="stSidebar"] .stFileUploader span {
        color: #333 !important;
        font-size: 0.95em;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Iris Classifier", layout="centered")

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# Sidebar: Model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Classifier",
    ("Random Forest", "SVM", "Logistic Regression")
)

# Sidebar: Hyperparameters
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, 10)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
elif model_choice == "SVM":
    c_value = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    model = SVC(probability=True, C=c_value, random_state=42)
else:
    max_iter = st.sidebar.slider("Max Iterations", 100, 500, 200, 50)
    model = LogisticRegression(max_iter=max_iter, random_state=42)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Calculate model evaluation metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
if model_choice == "Random Forest":
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

# Sidebar for single prediction input
st.sidebar.header("Input Features (Single Prediction)")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)
proba_df = pd.DataFrame(prediction_proba, columns=target_names)

# --- Improved Main Page UI Alignment ---

# Header with emoji and subtitle
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#4CAF50; font-size:3em;">🌸 Iris Species Prediction App</h1>
        <p style="font-size:1.2em; color:#555;">
            Predict the species of an iris flower using different classifiers.<br>
            Try single predictions, upload a CSV for batch predictions, or explore the data visually.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# First row: Input and Prediction side by side
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("#### Your Input")
    st.dataframe(pd.DataFrame(input_data, columns=df.columns[:-1]), use_container_width=True)

with col2:
    st.markdown("#### Prediction")
    st.success(f"The predicted species is: **{target_names[prediction[0]]}**")

# Add vertical space
st.markdown("<br>", unsafe_allow_html=True)

# Second row: Prediction Probabilities full width
st.markdown("#### Prediction Probabilities")
st.dataframe(proba_df, use_container_width=True)

st.markdown("---")

# Batch prediction section in a container
with st.container():
    st.header("📂 Batch Prediction (Upload CSV)")
    st.info("Upload a CSV file with columns: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        try:
            batch_pred = model.predict(batch_df)
            batch_pred_names = [target_names[i] for i in batch_pred]
            batch_df['Predicted Species'] = batch_pred_names
            st.dataframe(batch_df, use_container_width=True)
            # Download link
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "iris_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

st.markdown("---")

# Model evaluation in expandable sections
with st.expander("📊 Model Evaluation"):
    st.subheader("Model Accuracy on Test Set")
    st.info(f"Accuracy: {acc:.2f}")

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=target_names, columns=target_names), use_container_width=True)

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    if model_choice == "Random Forest":
        st.subheader("Feature Importances")
        st.bar_chart(imp_df.set_index('Feature'))

# Data Visualization Section
with st.expander("📈 Data Visualization"):
    st.markdown("#### Pairplot (Seaborn)")
    fig = sns.pairplot(df, hue="species", palette="Set1", diag_kind="hist")
    st.pyplot(fig)

    st.markdown("#### Feature Scatterplot")
    feature_x = st.selectbox("X-axis feature", df.columns[:-1], index=0, key="x")
    feature_y = st.selectbox("Y-axis feature", df.columns[:-1], index=1, key="y")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='species', palette='Set1', ax=ax2)
    st.pyplot(fig2)

# After model training and single prediction
if model_choice == "Random Forest":
    st.markdown("""
    <div class="shap-section">
    <h4>Prediction Explanation (SHAP Plot)</h4>
    """, unsafe_allow_html=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    class_idx = prediction[0]
    feature_names = list(df.columns[:-1])

    # Debug shapes
    st.markdown("<pre>", unsafe_allow_html=True)
    min_len = min(
        shap_values[class_idx][0].shape[0],
        input_data[0].shape[0],
        len(feature_names)
    )
    st.markdown("</pre>", unsafe_allow_html=True)

    force_plot_html = shap.force_plot(
        explainer.expected_value[class_idx],
        shap_values[class_idx][0][:min_len],
        input_data[0][:min_len],
        feature_names=feature_names[:min_len],
        show=False
    )
    custom_html = f"""
    <div style="background-color: white; padding: 10px; border-radius: 10px;">
        {shap.getjs()}
        {force_plot_html.html()}
    </div>
    """

    components.html(custom_html, height=400)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#888;'>
        Made with ❤️ using Streamlit and scikit-learn.<br>
        © Sameer017. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.header("Custom Training Data")
user_data_file = st.sidebar.file_uploader("Upload your own CSV for training (same columns as Iris)", type="csv")

if user_data_file is not None:
    user_df = pd.read_csv(user_data_file)
    # Assume last column is the target
    X = user_df.iloc[:, :-1]
    y = user_df.iloc[:, -1]
    # If target is not numeric, encode it
    if y.dtype == object:
        y, target_names = pd.factorize(y)
    else:
        target_names = np.unique(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    df = user_df.copy()
else:
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['species'], test_size=0.2, random_state=42)
