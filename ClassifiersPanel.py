import Visualization
import streamlit as st
import os

def PanelSetup(X, y, classifier_name, classifier_model, params, dataset_name):
	if classifier_name == "KNN":
		Visualization.visualize(X[:, :2],y, classifier_model, params, dataset_name)
		st.divider()
		knn_explanation =  st.button("What is KNN?")
		if knn_explanation:
			image_path = "./images/KNN.png"
			if os.path.exists(image_path):
				st.write("""
		### K-Nearest Neighbors Algorithm

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
	### Support Vector Machine Algorithm

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
	### Random Forest Algorithm

	### Training
	Random Forest constructs multiple decision trees during training. Each tree is trained on a random subset of the training data and a random subset of the features. This randomness helps in reducing overfitting and improving generalization.

	### Prediction
	When predicting the class (classification) or value (regression) of a new data point, Random Forest aggregates the predictions of all the individual trees in the forest. For classification, it uses a voting mechanism to determine the final class, while for regression, it calculates the mean prediction of all trees.

	### Parameters
	- **n_estimators**: The number of trees in the forest. A higher number of trees typically leads to better performance, but it also increases computational complexity.
	- **max_depth**: The maximum depth of each decision tree in the forest. Increasing max_depth allows the trees to learn more complex patterns in the data, but it may also lead to overfitting if set too high.

	### Classification
	In classification tasks, Random Forest assigns the class label that is most frequently predicted by the individual trees in the forest. This ensemble approach often results in improved accuracy and robustness compared to single decision trees.
	""")
				st.image(image_path, caption='Random Forest Classifier', use_column_width=True)
			else:
				st.write("Image not found. Please provide the correct image path.")
