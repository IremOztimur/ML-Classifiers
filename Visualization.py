import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np

cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

def plot_knn_decision_boundary(params, X, y, classifier_model, dataset_name):
    fig, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        classifier_model,
        X,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=params[dataset_name].feature_names[0],
        ylabel=params[dataset_name].feature_names[1],
        shading="auto",
    )
    # Plot training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=params[dataset_name].target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
        ax=ax,
    )
    plt.title("3-Class classification (k = %i)" % (params['K']))
    st.pyplot(fig)



def visualize(X, y, classifier_model, params, dataset_name, classifier_name):
	if (dataset_name == 'Digits'):
		st.write("Not available for Digits dataset")
		return
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	classifier_model.fit(X_train, y_train)
	y_pred = classifier_model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)

	if classifier_name == "KNN":
		st.write("## K-NN with {} Features".format(X.shape[1]))
	elif classifier_name == "SVM":
		st.write("## SVM with {} Features".format(X.shape[1]))

	col1, col2 = st.columns(2)
	col1.metric("Accuracy", accuracy)
	col2.metric("Number of classes", len(np.unique(y)))
	st.write("Shape of dataset: ", X.shape)

	if classifier_name == "KNN":
		plot_knn_decision_boundary(params, X, y, classifier_model, dataset_name)
	elif classifier_name == "SVM":
		pass


def plot_data(X_projected, y):
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    return plt.gcf()

