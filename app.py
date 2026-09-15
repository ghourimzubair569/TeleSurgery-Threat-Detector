
# Streamlit TeleSurgery Threat Detector
# Run with:
# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="TeleSurgery Threat Detector",
    page_icon="🔒"
)

st.title("🔒 TeleSurgery Threat Detector — Interactive UI")

st.markdown(
    """
    Upload your cybersecurity dataset in CSV format or use the example
    dataset available in the application folder.

    The application provides:

    - Dataset preview
    - Correlation heatmap
    - Mutual-information feature selection
    - Random Forest and Logistic Regression
    - Accuracy and classification metrics
    - Confusion matrix
    - ROC curve
    - Feature importance
    - Single-sample threat prediction
    - Threat severity
    - Security recommendations
    - Threat mitigation action plan
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Data / Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    use_example = st.checkbox(
        "Use example dataset shipped with app"
    )

    test_size = st.slider(
        "Test set size (%)",
        min_value=10,
        max_value=50,
        value=25
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        value=42,
        step=1
    )


# ============================================================
# LOAD DATA
# ============================================================

data = None


# Uploaded CSV has highest priority
if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        st.success(
            f"Uploaded dataset loaded successfully: "
            f"{uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"Could not read uploaded CSV: {e}"
        )

        st.stop()


# Example dataset
elif use_example:

    example_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "telesurgery_cybersecurity_dataset.csv"
    )

    if os.path.exists(example_path):

        try:

            data = pd.read_csv(
                example_path
            )

            st.success(
                "Example dataset loaded successfully."
            )

        except Exception as e:

            st.error(
                f"Example dataset could not be loaded: {e}"
            )

            st.stop()

    else:

        st.warning(
            """
            Example dataset was not found.

            Place:

            `telesurgery_cybersecurity_dataset.csv`

            in the same folder as `app.py`.

            Or upload your CSV using the uploader above.
            """
        )

        st.stop()


# Nothing selected
else:

    st.info(
        """
        👈 Please upload a CSV dataset from the sidebar.

        Your CSV should contain a target column named:

        **Threat Detected**
        """
    )

    st.stop()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("📋 Dataset Preview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Rows",
    data.shape[0]
)

col2.metric(
    "Columns",
    data.shape[1]
)

col3.metric(
    "Missing Values",
    int(data.isnull().sum().sum())
)

st.dataframe(
    data.head(10),
    use_container_width=True
)


# ============================================================
# BASIC CHECKS
# ============================================================

target = "Threat Detected"


if target not in data.columns:

    st.error(
        f"""
        ❌ Target column `{target}` was not found.

        Your dataset contains:

        {list(data.columns)}

        Please rename your target column to:

        `Threat Detected`
        """
    )

    st.stop()


# Remove completely empty columns
data = data.dropna(
    axis=1,
    how="all"
)


# Remove rows where target is missing
data = data.dropna(
    subset=[target]
).reset_index(
    drop=True
)


if data.shape[0] < 10:

    st.error(
        "The dataset has too few rows for reliable model training."
    )

    st.stop()


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

features = [
    c for c in data.columns
    if c != target
]


if len(features) == 0:

    st.error(
        "No feature columns were found in the dataset."
    )

    st.stop()


X = data[features].copy()

y = data[target].copy()


# ============================================================
# TARGET ENCODING
# ============================================================

target_encoder = None


if (
    y.dtype == "object"
    or str(y.dtype) == "category"
):

    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        y.astype(str)
    )

else:

    try:

        y = pd.to_numeric(y)

    except Exception:

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y.astype(str)
        )


# Convert target to numpy array
y = np.asarray(y)


# ============================================================
# CHECK TARGET CLASSES
# ============================================================

unique_classes = np.unique(y)


if len(unique_classes) != 2:

    st.error(
        f"""
        This application currently requires a binary target.

        Found {len(unique_classes)} classes:

        {unique_classes}
        """
    )

    st.stop()


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

label_encoders = {}


categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns


for col in categorical_columns:

    le = LabelEncoder()

    X[col] = X[col].fillna(
        "Unknown"
    ).astype(str)

    X[col] = le.fit_transform(
        X[col]
    )

    label_encoders[col] = le


# ============================================================
# CONVERT NUMERIC FEATURES
# ============================================================

for col in X.columns:

    if not pd.api.types.is_numeric_dtype(
        X[col]
    ):

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


# Replace infinity
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values
X = X.fillna(
    X.median(numeric_only=True)
)


# Remaining NaN values
X = X.fillna(0)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.subheader(
    "📈 Correlation Heatmap"
)


with st.expander(
    "Show correlation matrix and heatmap"
):

    corr = X.corr()

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Feature Correlation Matrix"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# FEATURE SELECTION
# ============================================================

st.subheader(
    "🎯 Feature Selection — Mutual Information"
)


max_features = len(
    X.columns
)


k_features = st.slider(
    "Select top-K features",
    min_value=1,
    max_value=max_features,
    value=min(8, max_features)
)


try:

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=k_features
    )

    selector.fit(
        X,
        y
    )

    scores = pd.Series(
        selector.scores_,
        index=X.columns
    ).sort_values(
        ascending=False
    )

except Exception as e:

    st.warning(
        f"Feature selection encountered an issue: {e}"
    )

    scores = pd.Series(
        np.ones(
            len(X.columns)
        ),
        index=X.columns
    ).sort_values(
        ascending=False
    )


selected_features = list(
    scores.head(
        k_features
    ).index
)


st.write(
    "### Top selected features"
)


st.dataframe(
    scores.head(
        k_features
    ).rename(
        "Mutual Information Score"
    )
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

st.subheader(
    "🤖 Train & Evaluate Models"
)


X_sel = X[
    selected_features
]


try:

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y,
        test_size=test_size / 100.0,
        random_state=int(random_state),
        stratify=y
    )

except Exception:

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y,
        test_size=test_size / 100.0,
        random_state=int(random_state)
    )


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=int(random_state),
    class_weight="balanced"
)


rf.fit(
    X_train,
    y_train
)


rf_pred = rf.predict(
    X_test
)


rf_acc = accuracy_score(
    y_test,
    rf_pred
)


# Probability of positive class
rf_proba_all = rf.predict_proba(
    X_test
)


if rf_proba_all.shape[1] == 2:

    rf_proba = rf_proba_all[:, 1]

else:

    rf_proba = rf_proba_all[:, 0]


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

log = LogisticRegression(
    max_iter=1000
)


log.fit(
    X_train,
    y_train
)


log_pred = log.predict(
    X_test
)


log_acc = accuracy_score(
    y_test,
    log_pred
)


# ============================================================
# MODEL METRICS
# ============================================================

col1, col2, col3 = st.columns(3)


col1.metric(
    "🌲 Random Forest Accuracy",
    f"{rf_acc * 100:.2f}%"
)


col2.metric(
    "📉 Logistic Regression Accuracy",
    f"{log_acc * 100:.2f}%"
)


col3.metric(
    "🧪 Test Samples",
    len(X_test)
)


# ============================================================
# DETAILED METRICS
# ============================================================

with st.expander(
    "📊 Detailed Metrics & Plots"
):

    st.write(
        "### Random Forest — Classification Report"
    )

    st.text(
        classification_report(
            y_test,
            rf_pred,
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.write(
        "### Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        rf_pred
    )


    fig, ax = plt.subplots()


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.write(
        "### ROC Curve"
    )


    try:

        rf_auc = roc_auc_score(
            y_test,
            rf_proba
        )


        fpr, tpr, _ = roc_curve(
            y_test,
            rf_proba
        )


        fig, ax = plt.subplots()


        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {rf_auc:.3f}"
        )


        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )


        ax.set_title(
            "Random Forest ROC Curve"
        )


        ax.set_xlabel(
            "False Positive Rate"
        )


        ax.set_ylabel(
            "True Positive Rate"
        )


        ax.legend()


        st.pyplot(
            fig,
            use_container_width=True
        )


        plt.close(fig)


    except Exception as e:

        st.info(
            f"ROC curve could not be calculated: {e}"
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.write(
        "### Feature Importances"
    )


    fi = pd.Series(
        rf.feature_importances_,
        index=X_sel.columns
    ).sort_values(
        ascending=True
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                4,
                len(fi) * 0.4
            )
        )
    )


    fi.plot(
        kind="barh",
        ax=ax
    )


    ax.set_xlabel(
        "Importance"
    )


    ax.set_ylabel(
        "Feature"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


# ============================================================
# SINGLE SAMPLE PREDICTION
# ============================================================

st.subheader(
    "🔍 Predict a Single Sample"
)


st.write(
    """
    Enter values for the selected features below.

    The Random Forest model will estimate whether a
    cybersecurity threat is detected.
    """
)


with st.form(
    "single_predict"
):

    input_data = {}


    for col in X_sel.columns:

        dtype = X[col].dtype


        # ----------------------------------------------------
        # CATEGORICAL FEATURE
        # ----------------------------------------------------

        if col in label_encoders:

            categories = (
                label_encoders[col]
                .classes_
                .tolist()
            )


            choice = st.selectbox(
                col,
                categories
            )


            input_data[col] = (
                label_encoders[col]
                .transform(
                    [choice]
                )[0]
            )


        # ----------------------------------------------------
        # INTEGER FEATURE
        # ----------------------------------------------------

        elif np.issubdtype(
            dtype,
            np.integer
        ):

            median_value = X[col].median()


            if pd.isna(
                median_value
            ):

                median_value = 0


            input_data[col] = st.number_input(
                col,
                value=int(
                    median_value
                )
            )


        # ----------------------------------------------------
        # FLOAT FEATURE
        # ----------------------------------------------------

        else:

            median_value = X[col].median()


            if pd.isna(
                median_value
            ):

                median_value = 0.0


            input_data[col] = st.number_input(
                col,
                value=float(
                    median_value
                )
            )


    submitted = st.form_submit_button(
        "🚨 Predict Threat"
    )


# ============================================================
# PREDICTION VARIABLES
# ============================================================

# IMPORTANT:
# These variables are initialized BEFORE the button condition.
# This prevents NameError when the page first loads.

rf_prediction = None
threat_probability = None
severity = None
label = None


# ============================================================
# RUN PREDICTION
# ============================================================

if submitted:

    sample_df = pd.DataFrame(
        [input_data]
    )


    # Make sure columns have correct order
    sample_df = sample_df[
        X_sel.columns
    ]


    # Random Forest prediction
    rf_prediction = rf.predict(
        sample_df
    )[0]


    # Probability
    probability_values = rf.predict_proba(
        sample_df
    )[0]


    # Probability of class 1
    if len(
        probability_values
    ) == 2:

        threat_probability = float(
            probability_values[1]
        )

    else:

        threat_probability = float(
            probability_values[0]
        )


    # --------------------------------------------------------
    # THREAT LABEL
    # --------------------------------------------------------

    if rf_prediction == 1:

        label = "Yes"

    else:

        label = "No"


    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    if rf_prediction == 0:

        severity = "None"

    else:

        if threat_probability >= 0.85:

            severity = "High"

        elif threat_probability >= 0.60:

            severity = "Medium"

        else:

            severity = "Low"


    # --------------------------------------------------------
    # DETECTION RESULT
    # --------------------------------------------------------

    st.write("---")

    st.write(
        "## 🛡️ Detection Result"
    )


    result_col1, result_col2, result_col3 = st.columns(3)


    result_col1.metric(
        "Threat Detected",
        label
    )


    result_col2.metric(
        "Threat Probability",
        f"{threat_probability * 100:.2f}%"
    )


    result_col3.metric(
        "Severity",
        severity
    )


    # ========================================================
    # SECURITY RECOMMENDATIONS
    # ========================================================

    st.write("---")

    st.subheader(
        "🛡️ Security Recommendations"
    )


    if rf_prediction == 0:

        st.success(
            "✅ No immediate cybersecurity threat was detected."
        )


        suggestions = [
            "Continue real-time monitoring of the tele-surgery network.",
            "Keep authentication and access-control policies enabled.",
            "Maintain multi-factor authentication for remote operators.",
            "Keep the surgical workstation, server, and network devices patched.",
            "Regularly review security logs for unusual activity.",
            "Continue periodic vulnerability assessments and security testing.",
        ]


    else:

        # ----------------------------------------------------
        # HIGH RISK
        # ----------------------------------------------------

        if severity == "High":

            st.error(
                "🚨 HIGH-RISK THREAT DETECTED — Immediate action is recommended."
            )


            suggestions = [
                "Isolate the affected surgical workstation or network segment from other systems.",
                "Temporarily suspend remote-control access until the incident has been investigated.",
                "Verify the identity and authorization of the remote operator.",
                "Re-authenticate users and reset credentials that may have been exposed.",
                "Review authentication logs for repeated failures or suspicious login attempts.",
                "Review command logs and investigate unexpected or unauthorized commands.",
                "Check network traffic for unusual connections or communication with unknown systems.",
                "Apply the latest security patches after safe validation.",
                "Preserve system, authentication, network, and application logs for forensic investigation.",
                "Escalate the incident to the organization's cybersecurity or incident-response team.",
            ]


        # ----------------------------------------------------
        # MEDIUM RISK
        # ----------------------------------------------------

        elif severity == "Medium":

            st.warning(
                "⚠️ MEDIUM-RISK THREAT DETECTED — Investigate and strengthen security controls."
            )


            suggestions = [
                "Increase monitoring of the affected tele-surgery connection.",
                "Verify the remote operator's identity and authorization.",
                "Force re-authentication for the affected session.",
                "Review recent commands for unexpected or unauthorized activity.",
                "Check network latency, packet loss, and jitter for abnormal behavior.",
                "Review authentication failures and investigate unusual login patterns.",
                "Restrict unnecessary remote access until the investigation is complete.",
                "Apply available security updates after testing.",
                "Continue enhanced monitoring for repeated suspicious activity.",
            ]


        # ----------------------------------------------------
        # LOW RISK
        # ----------------------------------------------------

        else:

            st.info(
                "ℹ️ LOW-RISK ANOMALY DETECTED — Continue monitoring and investigate."
            )


            suggestions = [
                "Review the detected anomaly and compare it with normal system behavior.",
                "Verify the remote operator's identity.",
                "Check authentication logs for unusual activity.",
                "Review network latency, packet loss, and jitter.",
                "Monitor the session for repeated suspicious behavior.",
                "Ensure multi-factor authentication is enabled.",
                "Keep the tele-surgery system and security software updated.",
            ]


    # --------------------------------------------------------
    # DISPLAY SUGGESTIONS
    # --------------------------------------------------------

    st.write(
        "### 🔐 Recommended Prevention / Mitigation"
    )


    for suggestion in suggestions:

        st.markdown(
            f"- {suggestion}"
        )


    # ========================================================
    # SECURITY ACTION PLAN
    # ========================================================

    if rf_prediction == 1:

        st.write("---")

        st.write(
            "### 🔐 Recommended Security Action Plan"
        )


        action_col1, action_col2, action_col3 = st.columns(3)


        action_col1.info(
            """
            **STEP 1 — CONTAIN**

            Limit or isolate the affected connection
            and prevent further unauthorized access.
            """
        )


        action_col2.info(
            """
            **STEP 2 — INVESTIGATE**

            Review authentication, network, command,
            and system logs to identify the cause.
            """
        )


        action_col3.info(
            """
            **STEP 3 — RECOVER**

            Patch affected systems, restore secure access,
            reset credentials, and continue monitoring.
            """
        )


        st.caption(
            "These recommendations are general cybersecurity guidance "
            "for a research/demo system and should be adapted to the "
            "organization's validated tele-surgery incident-response procedures."
        )


# ============================================================
# DOWNLOAD MODEL
# ============================================================

st.write("---")

st.subheader(
    "💾 Export Model"
)


if st.button(
    "Download trained Random Forest model"
):

    model_package = {
        "model": rf,
        "selected_features": selected_features,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder
    }


    buffer = io.BytesIO()


    pickle.dump(
        model_package,
        buffer
    )


    buffer.seek(0)


    st.download_button(
        label="⬇️ Click to Download Model",
        data=buffer,
        file_name="telesurgery_rf_model.pkl",
        mime="application/octet-stream"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "TeleSurgery Threat Detector — "
    "Machine-learning based cybersecurity analysis and threat detection."
)

