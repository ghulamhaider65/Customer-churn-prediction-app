import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from io import StringIO

np.random.seed(42)


def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn2.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df = df.drop(columns=['customerID'])
    return df


def preprocess_data(df):
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    object_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'PaymentMethod']

    dummy_frames = []
    for col in object_cols:
        dummies = pd.get_dummies(df[col], prefix=col, dtype='int64')
        dummy_frames.append(dummies)

    df = df.drop(columns=object_cols)
    df_encoded = pd.concat([df] + dummy_frames, axis=1)

    return df_encoded


def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def feature_selection(X, y, k=15):
    cols_to_drop = [
        'StreamingMovies_No internet service',
        'StreamingTV_No internet service',
        'TechSupport_No internet service',
        'DeviceProtection_No internet service',
        'OnlineBackup_No internet service',
        'OnlineSecurity_No internet service'
    ]
    cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    X_filtered = X.drop(columns=cols_to_drop, errors='ignore')

    selector = SelectKBest(score_func=f_classif, k=min(k, len(X_filtered.columns)))
    X_selected = selector.fit_transform(X_filtered, y)
    selected_features = X_filtered.columns[selector.get_support()]

    return X_filtered[selected_features], selected_features


def train_model(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        early_stopping_rounds=10,
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return grid_search.best_estimator_


def save_model(model, selected_features, filename='churn_model.pkl'):
    joblib.dump({
        'model': model,
        'features': selected_features,
        'feature_importances': dict(zip(selected_features, model.feature_importances_))
    }, filename)


def load_model(filename='churn_model.pkl'):
    return joblib.load(filename)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": feature_importance
    }


def predict_single(model, input_df):
    return model.predict(input_df), model.predict_proba(input_df)[:, 1]


def predict_bulk(model, file_obj, selected_features):
    if isinstance(file_obj, str):
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_csv(file_obj)

    df_processed = preprocess_data(df)

    missing_features = set(selected_features) - set(df_processed.columns)
    for feature in missing_features:
        df_processed[feature] = 0

    df_processed = df_processed[selected_features]

    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    return df_processed, predictions, probabilities


def get_feature_names():
    df = load_data()
    df_encoded = preprocess_data(df)
    return df_encoded.drop('Churn', axis=1).columns.tolist()



