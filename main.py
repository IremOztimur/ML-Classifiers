import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title("Classifiers")

st.divider()

st.sidebar.write("## Explore different classifiers")
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Wine dataset":
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)

def add_parameter_ui(clf_name):
    params = dict()
    if (clf_name == "KNN"):
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif (clf_name == "SVM"):
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif (clf_name == "Random Forest"):
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
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


st.write(f"## Classifier - {classifier_name}")
col1, col2 = st.columns(2)
col1.metric("Accuracy", accuracy)
col2.metric("Number of classes", len(np.unique(y)))
st.write("Shape of dataset: ", X.shape)

st.divider()

if classifier_name == "KNN":
	knn_explanation =  st.button("What is KNN?")
	if knn_explanation:
		image_path = "./images/KNN.png"
		if os.path.exists(image_path):
			st.write("""
	### Training
	KNN stores all available data points and their corresponding class labels (for classification) or values (for regression).

	### Prediction
	When predicting the class of a new data point, KNN calculates the distances between the new data point and all other data points in the training set.

	### Choosing K
	K is a hyperparameter representing the number of nearest neighbors to consider. The algorithm selects the K nearest data points based on their distances to the new data point.

	### Classification
	For classification tasks, KNN assigns the class label that is most common among the K nearest neighbors to the new data point.
	""")
			st.image(image_path, caption='K-NN (K-Nearest Neighbors) Algorithm', use_column_width=True)
			st.write("The example above uses K equal to 3. The algorithm calculates the probability of each class among these neighbors. For instance, if 2 out of the 3 nearest neighbors are green and 1 is red, the probability of the new point belonging to the green class is higher. Thus, the algorithm assigns the new data point to the green class.")
		else:
			st.write("Image not found. Please provide the correct image path.")

if classifier_name == "SVM":
	svm_explanation =  st.button("What is SVM?")
	if svm_explanation:
		image_path = "./images/SVM.png"
		if os.path.exists(image_path):
			st.write("""
### Training
Support Vector Machine (SVM) learns to identify the optimal hyperplane that best separates data points of different classes in the training set. This hyperplane is chosen to maximize the margin, the distance between the hyperplane and the nearest data points of each class, known as support vectors.

### Prediction
When predicting the class of a new data point, SVM evaluates which side of the hyperplane the point falls on. Points on one side of the hyperplane are classified as one class, while points on the other side are classified as the other class.

### Kernel Trick
SVMs can handle non-linearly separable data by using kernel functions to map the data into higher-dimensional spaces, where it becomes linearly separable. Common kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid.

### Sparsity
SVMs often utilize only a subset of the training data points, called support vectors, to define the decision boundary. This sparsity property makes SVMs memory-efficient and effective for high-dimensional data.

### Classification and Regression
SVMs can be used for both classification and regression tasks. In classification, SVMs aim to find the hyperplane that best separates different classes, while in regression, they aim to find the hyperplane that best fits the data points while minimizing error.

### C Parameter in SVM
The "C" parameter in Support Vector Machine (SVM) is a regularization parameter used to control the trade-off between maximizing the margin and minimizing the classification error.

### Margin and Error Trade-off
- **High C Value**: A high value of C results in a smaller margin but less misclassification. In other words, the SVM classifier tries to classify all training examples correctly by allowing fewer misclassifications.
- **Low C Value**: A low value of C results in a larger margin but more misclassification. In this case, the SVM classifier prioritizes maximizing the margin, even if it leads to some misclassification.

### Overfitting and Underfitting
- A very high value of C may lead to overfitting, where the model learns the training data too well but may not generalize well to unseen data.
- Conversely, a very low value of C may lead to underfitting, where the model fails to capture the underlying patterns in the data.

### Choosing the Right C Value
The choice of the C parameter depends on the specific dataset and problem at hand. It is often determined using techniques such as cross-validation, where different values of C are tried, and the one resulting in the best performance on a validation set is selected.

In summary, the C parameter in SVM allows you to control the balance between the complexity of the model and its ability to generalize to unseen data.

			""")
			st.image(image_path, caption='Support Vector Machine (SVM)', use_column_width=True)
		else:
			st.write("Image not found. Please provide the correct image path.")

if classifier_name == "Random Forest":
	randomf_explanation =  st.button("What is Random Forest?")
	if randomf_explanation:
		image_path = "./images/Random-Forest.png"
		if os.path.exists(image_path):
			st.write("""
			""")

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

def plot_data(X_projected, y):
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    return plt.gcf()

st.divider()

fig = plot_data(X_projected, y)
plt.show()
st.write("### 2D Dataset Visualization with PCA")
st.pyplot(fig)
