from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import streamlit as st
import pandas as pd
import joblib
import time

df = pd.read_csv("datasets/machine_learning/synthetic_dataset.csv")

#Delete column where nan happends
df = df.dropna(subset = ['Stock'])
df['Stock'] = df['Stock'].map({'In Stock': 1, 'Out of Stock': 0})

X = df.drop(columns = ['Stock'])
y = df['Stock']

numerical_features = ['Price', 'Rating', 'Discount']
category_features = ['Category']

numerical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

category_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'Unknown')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocesser = ColumnTransformer(transformers = [
    ('numerical', numerical_transformer, numerical_features),
    ('category', category_transformer, category_features)
])

model_1 = RandomForestClassifier(n_estimators = 100, random_state = 42)
model_2 = HistGradientBoostingClassifier(random_state = 42)
model_3 = LogisticRegression()

ensemble = VotingClassifier(estimators = [
    ('RandomForest', model_1),
    ('HistG', model_2),
    ('LogisticR', model_3)
],
    voting = 'soft'
)

clf = Pipeline(steps = [
    ('Preprocesor', preprocesser),
    ('Ensemble', ensemble)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf.fit(X_train, y_train)

print(f"Ensemble acc:  {clf.score(X_test, y_test):.4f}")
joblib.dump(clf, 'models/ensemble/ensemble_model.pkl')

#Display to website
st.title("Product Stock Predictor")

form_values = {
    "price": None,
    "rating": None,
    "discount": None,
    "category": None
}

with st.form(key = "user_input_form"):
    form_values["price"] = st.number_input("Enter price")
    form_values["rating"]  = st.number_input("Enter rating")
    form_values["discount"] = st.number_input("Enter discount")
    form_values["category"] = st.selectbox("Category", ['A', 'B', 'C', 'D', 'Unknown'])
    submit_button = st.form_submit_button(label = "Predict!")

    if submit_button:
        model = joblib.load('models/ensemble/ensemble_model.pkl')
        input_df = pd.DataFrame([[form_values["category"], form_values["price"], form_values["rating"], form_values["discount"]]], 
                                columns = ['Category', 'Price', 'Rating', 'Discount'])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.success("This product is likely in stock.")
        else:
            st.error("This product is likely out of stock.")

        percent = "My prediction confidence: " + f'{confidence:.2f}'
        confidence_bar = st.progress(0, text = percent)
        for percent_complete in range(int(confidence)):
            time.sleep(0.01)
            confidence_bar.progress(percent_complete + 1, text = percent)