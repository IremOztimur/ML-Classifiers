import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import ClassifiersPanel
import Visualization


st.title("Classifiers")

st.divider()

st.sidebar.write("## Explore different classifiers")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset", "Digits"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine dataset":
        data = datasets.load_wine()
    elif dataset_name == "Digits":
          data = datasets.load_digits()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

def add_parameter_ui(clf_name):
    params = dict()
    if (clf_name == "KNN"):
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
        params[dataset_name] = datasets.load_iris()
    elif (clf_name == "SVM"):
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
        params[dataset_name] = datasets.load_iris()
    elif (clf_name == "Random Forest"):
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params[dataset_name] = datasets.load_iris()
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if (clf_name == "KNN"):
        clf_model = KNeighborsClassifier(n_neighbors=params["K"])
    elif (clf_name == "SVM"):
        clf_model = SVC(C=params["C"])
    elif (clf_name == "Random Forest"):
        clf_model = RandomForestClassifier(max_depth=params["max_depth"],
                                      n_estimators=params["n_estimators"],
                                      random_state=42)
    return clf_model

classifier_model = get_classifier(classifier_name, params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier_model.fit(X_train, y_train)

y_pred = classifier_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


st.write(f"## Classifier - {classifier_name} with {X.shape[1]} features")
col1, col2 = st.columns(2)
col1.metric("Accuracy", accuracy)
col2.metric("Number of classes", len(np.unique(y)))
st.write("Shape of dataset: ", X.shape)

st.divider()

ClassifiersPanel.PanelSetup(X, y, classifier_name, classifier_model, params, dataset_name)

st.sidebar.divider()

pca_explanation = st.sidebar.button("What is PCA?")

if pca_explanation:
    pca_text = "<b>Principal Component Analysis (PCA)</b> is a technique used to simplify complex datasets by reducing their dimensionality while retaining important information. It transforms the original features into a new set of orthogonal components called principal components, ordered by their variance. PCA helps in visualizing high-dimensional data and speeding up machine learning algorithms by removing redundant or noisy features."
    st.sidebar.write(pca_text, unsafe_allow_html=True)
    image_path = "./images/PCA.png"
    if os.path.exists(image_path):
         st.sidebar.image(image_path, caption="PCA ( Principal Component Analysis)", use_column_width=True)


#PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)

st.divider()

fig = Visualization.plot_data(X_projected, y)

st.write("### 2D Dataset Visualization with PCA")
st.pyplot(fig)
