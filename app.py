import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml_pipeline import (
    load_data, preprocess_data, feature_selection,
    train_model, evaluate_model, save_model, load_model, predict_bulk, predict_single
)
import os
import base64

# Loading CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def plot_feature_importance(model, features):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance', color='Importance',
                 color_continuous_scale='Bluered')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_churn_distribution(df):
    fig = px.pie(df, names='Churn', title='Churn Distribution',
                 color='Churn', color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'})
    st.plotly_chart(fig, use_container_width=True)

def plot_numerical_distribution(df, column):
    fig = px.histogram(df, x=column, color='Churn',
                       marginal='box', barmode='overlay',
                       color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'})
    st.plotly_chart(fig, use_container_width=True)

def plot_categorical_distribution(df, column):
    fig = px.histogram(df, x=column, color='Churn', barmode='group',
                       color_discrete_map={'No': '#4CAF50', 'Yes': '#F44336'})
    st.plotly_chart(fig, use_container_width=True)

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">Download Predictions as CSV</a>'

def main():
    st.subheader("📊 Telco Customer Churn Prediction")
    st.markdown("Predict which customers are likely to churn and visualize key insights")

    menu = ["Train Model", "Single Prediction", "Bulk Prediction", "Data Exploration"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Train Model":
        st.subheader("Train the Churn Prediction Model")
        st.info("This will train a new XGBoost model on the Telco customer data")

        df = load_data()
        df_encoded = preprocess_data(df)
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        X_selected, selected_features = feature_selection(X, y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        if st.button('Train Model'):
            with st.spinner('Training model... This may take a few minutes'):
                model = train_model(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                save_model(model, selected_features, 'churn_model.pkl')
                st.success('Model trained and saved successfully!')

                st.write("## Model Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                col2.metric("Precision", f"{metrics['precision']:.2%}")
                col3.metric("Recall", f"{metrics['recall']:.2%}")

                st.write("### Confusion Matrix")
                fig = px.imshow(metrics['conf_matrix'],
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['No Churn', 'Churn'],
                                y=['No Churn', 'Churn'],
                                text_auto=True,
                                color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)

                st.write("### Feature Importance")
                plot_feature_importance(model, selected_features)

    elif choice == "Single Prediction":
        st.subheader("Predict Churn for a Single Customer")
        saved = load_model("churn_model.pkl")
        model = saved['model']
        selected_features = saved['features']

        with st.expander("Enter Customer Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])

            with col2:
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        input_data = {
            'gender': 1 if gender == "Male" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
            'MonthlyCharges': monthly_charges,
            'InternetService_Fiber optic': int(internet_service == "Fiber optic"),
            'InternetService_No': int(internet_service == "No"),
            'PaymentMethod_Electronic check': int(payment_method == "Electronic check"),
            'PaymentMethod_Mailed check': int(payment_method == "Mailed check"),
            'PaymentMethod_Bank transfer (automatic)': int(payment_method == "Bank transfer (automatic)"),
            'PaymentMethod_Credit card (automatic)': int(payment_method == "Credit card (automatic)")
        }

        input_df = pd.DataFrame([input_data])
        for col in selected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[selected_features]

        if st.button("Predict Churn"):
            prediction, prob = predict_single(model, input_df)
            churn_status = "High" if prediction[0] == 1 else "Low"
            color = "red" if prediction[0] == 1 else "green"
            emoji = "🔴" if prediction[0] == 1 else "🟢"

            st.markdown(f"""
            <div class="churn-prediction churn-{'yes' if prediction[0] == 1 else 'no'}">
                <h3>{emoji} {churn_status} Churn Risk</h3>
                <p>This customer has a <b>{prob[0] * 100:.1f}%</b> probability of churning.</p>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob[0] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    elif choice == "Bulk Prediction":
        st.subheader("Predict Churn for Multiple Customers")
        saved = load_model("churn_model.pkl")
        model = saved['model']
        selected_features = saved['features']

        st.info("Upload a CSV file with customer data. Ensure it has the same structure as the training data.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.write(df.head())

            if st.button("Predict for All Customers"):
                with st.spinner('Processing predictions...'):
                    df_processed = preprocess_data(df)
                    missing_features = set(selected_features) - set(df_processed.columns)
                    for feature in missing_features:
                        df_processed[feature] = 0

                    predictions, probabilities = predict_bulk(model, uploaded_file, selected_features)
                    result_df = df.copy()
                    result_df['Churn Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                    result_df['Churn Probability'] = probabilities

                    st.success("Predictions completed!")
                    st.write("### Prediction Results")
                    st.write(result_df)
                    st.metric("Predicted Churn Rate",
                             f"{result_df['Churn Prediction'].value_counts(normalize=True).get('Yes', 0):.1%}")
                    st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)
                    st.write("### Churn Distribution in Predictions")
                    plot_churn_distribution(result_df.rename(columns={'Churn Prediction': 'Churn'}))

    elif choice == "Data Exploration":
        st.subheader("Data Exploration and Insights")
        df = load_data()
        df_encoded = preprocess_data(df)

        st.write("### Dataset Overview")
        st.write(f"Total customers: {len(df)}")
        st.write(df.head())

        st.write("### Churn Distribution")
        plot_churn_distribution(df)

        st.write("### Numerical Features Analysis")
        num_feature = st.selectbox("Select numerical feature", ['MonthlyCharges', 'tenure'])
        plot_numerical_distribution(df, num_feature)

        st.write("### Categorical Features Analysis")
        cat_feature = st.selectbox("Select categorical feature",
                                 ['Contract', 'PaymentMethod', 'InternetService', 'gender'])
        plot_categorical_distribution(df, cat_feature)

        st.write("### Correlation Analysis")
        numeric_df = df_encoded.select_dtypes(include=['int64', 'float64'])
        fig = px.imshow(numeric_df.corr(), color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()


