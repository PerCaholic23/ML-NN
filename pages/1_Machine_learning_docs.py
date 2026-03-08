import streamlit as st

def paragraph(text, indent = 1):
    tab = "&nbsp;" * 4 * indent
    st.markdown(f"{tab}{text}", unsafe_allow_html = True)

st.title("*Data Preparation*")
paragraph("First and foremost, [this](https://www.kaggle.com/datasets/himelsarder/retail-product-dataset-with-missing-values) is where I got the dataset to build this model. " \
"Before training the model, the dataset needs to be *cleaned and prepared*.")
paragraph("Raw data usually contains missing values, inconsistent formats, or categories that machine learning models cannot directly understand. " \
"Because of that, several preprocessing steps are applied.")

st.subheader("Handling Missing Values")
paragraph("Missing values in numerical columns are replaced using the **median** of the column. The median is used instead of the mean because it is less affected by extreme values.")
paragraph("For categorical data such as product category, missing values are replaced with **Unknown**")

st.subheader("Scaling Numerical Features")
paragraph("The numerical features are standardized using **StandardScaler**. This process transforms the values so they have a similar scale. It helps some machine learning algorithms perform more consistently.")

st.subheader("Encoding Categories")
paragraph("Machine learning models cannot directly process text categories such as **A** or **B**. To solve this, One-Hot Encoding is used. This method converts each category into binary columns.")

st.markdown("<hr style='height:3px;border:none;background-color:gray;'>", unsafe_allow_html=True )

st.title("*Ensemble Algorithms*")
paragraph("This project uses 3 algorithm models combined togehter to make the final prediction. Using several models often improves prediction performance compared to using a single model.")

st.subheader("Random Forest")
paragraph("Random Forest is based on decision trees. Instead of building just one tree, it creates many trees using different random subsets of the data. Each tree makes its own prediction, and the final result is determined by combining the predictions from all trees.")
paragraph("This approach helps reduce overfitting and usually performs well on many types of datasets.")

st.subheader("Histogram Gradient Boosting")
paragraph("Histogram Gradient Boosting is an improved version of gradient boosting algorithms. It builds trees sequentially, where each new tree tries to correct the errors made by the previous trees. " \
"The histogram-based method groups feature values into bins, which makes training faster and more efficient.")

st.subheader("Logistic Regression")
paragraph("Logistic Regression is one of the simplest classification algorithms. It calculates the probability that an input belongs to a particular class." \
" The algorithm uses a mathematical function called the sigmoid function to convert predictions into probabilities between 0 and 1.")

st.markdown("<hr style='height:3px;border:none;background-color:gray;'>", unsafe_allow_html=True )

st.title("Model Development Steps")
st.markdown("#### 1st step: Load the dataset")
st.markdown("""
- The dataset is loaded using a data processing library and stored in a dataframe.
""")

st.markdown("#### 2nd step: Clean the data")
st.markdown("""
- Rows with missing values in the target column are removed, and the target labels are converted into numeric values.
""")

st.markdown("#### 3rd step: Define features")
st.markdown("""
- The input features and the target variable are separated.
""")

st.markdown("#### 4th step: Create machine learning models")
st.markdown("""
3 models are defined:
- Random Forest
- Histogram Gradient Boosting
- Logistic Regression
""")

st.markdown("#### 5th step: Combine models into an ensemble")
st.markdown("""
- The models are combined to improve prediction performance.
""")

st.markdown("#### 6th step: Split the data")
st.markdown("""
- The dataset is divided into training and testing sets so the model can be evaluated on unseen data.
""")

st.markdown("#### 7th step: Train & Evaluate the model")
st.markdown("""
- The pipeline is trained using the training data.
- The trained model is tested on the test dataset to measure its accuracy.
""")

st.markdown("#### 8th step: Save the model")
st.markdown("""
- The trained model is saved using joblib so it can be reused later without retraining.
""")

st.markdown("#### 9th step: Deploy the model")
st.markdown("""
- Finally, the model is integrated into a Streamlit web application where users can input product information and receive predictions about stock availability.
""")


st.markdown("<hr style='height:3px;border:none;background-color:gray;'>", unsafe_allow_html=True )

st.title("References")
st.markdown("These are the main resources I use to understand the algorithms and tools")
st.markdown("""
- [Retail Product Dataset with Missing Values](https://www.kaggle.com/datasets/himelsarder/retail-product-dataset-with-missing-values) as for training dataset
- [Olivier Grisel - Histogram-based Gradient Boosting in scikit-learn 0.21](https://www.youtube.com/watch?v=urVUlKbQfQ4) by EuroPython Conference
- [Logistic Regression (and why it's different from Linear Regression)](https://www.youtube.com/watch?v=3bvM3NyMiE0&t=153s) by Visually Explained
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8&t=189s) by StatQuest with Josh Starmer
- [Ensemble (Boosting, Bagging, and Stacking) in Machine Learning](https://www.youtube.com/watch?v=sN5ZcJLDMaE&t=38s) by Emma Ding
- [Ensemble learners](https://www.youtube.com/watch?v=Un9zObFjBH0&t=28s) by Udacity
- [Streamlit Mini Course - Make Websites With ONLY Python](https://www.youtube.com/watch?v=o8p7uQCGD0U&t=625s) by Tech With Tim
- [Streamlit basic concepts](https://docs.streamlit.io/get-started/fundamentals/main-concepts) as for building website
""")