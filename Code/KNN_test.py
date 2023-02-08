import numpy as np
import matplotlib.pyplot as plt
import KNN
from matplotlib.colors import ListedColormap
import plotly.express as px
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create color maps
cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff'])
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]

iris = datasets.load_iris()
X, y = iris.data, iris.target
# print(y)
Z = iris.get('feature_names')
# print(Z)
A = iris.get('target_names')
# print(A)

"""
X_train:training samples
X_test: test samples
y_train: training labels
y_test : test lables
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# shape of input--rows & columns - samples & lables
print(X_train.shape)
# print(X_train[0])  # features of first row

# shape of output-only one colum
print(y_train.shape)
# print(y_train)

# shape of input--rows & columns - samples & lables
print(X_test.shape)
# print(X_test[0])  # features of first row

# shape of output-only one colum
print(y_test.shape)
# print("Test", y_test)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.title("Trend between Sepal Width and Sepal Length and Target")
plt.xlabel("Sepal Width")
plt.ylabel("Sepal Length")
plt.show()

# classifier clf
clf = KNN.KNN(3)

# fit method training data
clf.fit(X_train, y_train)

#  predict test sample
predictions = clf.predict(X_test)

# test accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"My KNN Model Accuracy is :  {accuracy:.3}")

#  sklearn KNN
s = KNeighborsClassifier(n_neighbors=3)
s.fit(X_train, y_train)
predict = s.predict(X_test)
print(f"Scikit learn KNN classifier accuracy: {accuracy_score(y_test, predict):.3}")


def tune_parameter(X_train, y_train, X_test, y_test, k_num):
    accuracy = []
    # y_test = y_testtt
    for i in range(1, k_num):
        ms = KNeighborsClassifier(n_neighbors=i)
        # model_sklearn.fit(X, y)
        ms.fit(X_train, y_train)
        y_pred = ms.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
    return np.array(accuracy)


accuracy_1 = tune_parameter(X_train, y_train, X_test, y_test, 100)
# print("ACC",accuracy_1)

x = np.arange(1, 100)
y = accuracy_1
plt.title("Trend between Sepal Width and Sepal Length and Target")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.plot(x, y, color="blue")
plt.show()

fig = px.scatter(y_test, color=predictions)
fig.update_traces(marker_size=12, marker_line_width=1.5)
fig.update_layout(legend_orientation='h')
fig.show()

M = iris.data[:, :2]
N = iris.target
msk = KNeighborsClassifier(n_neighbors=3)
msk.fit(M, N)
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    msk,
    M,
    cmap=cmap_light,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
    shading="auto",
)

sns.scatterplot(
    x=M[:, 0],
    y=M[:, 1],
    hue=iris.target_names[N],
    palette=cmap_bold,
    alpha=1.0,
    edgecolor="black",
)

plt.show()

error_rate = []  # list that will store the average error rate value of k
for i in range(1, 31):  # Took the range of k from 1 to 50
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train, y_train)
    predict_i = clf.predict(X_test)
    error_rate.append(np.mean(predict_i != y_test))
error_rate

# plotting the error rate vs k graph
plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), error_rate, marker="o", markerfacecolor="green",
         linestyle="dashed", color="red", markersize=15)
plt.title("Error rate vs k value", fontsize=20)
plt.xlabel("k- values", fontsize=20)
plt.ylabel("error rate", fontsize=20)
plt.xticks(range(1, 31))
plt.show()

error = [1 - x for x in error_rate]
optimal_n = error.index(min(error))
print("The optimal K value from the graph is :", optimal_n)

m = KNeighborsClassifier(n_neighbors=optimal_n)
m.fit(X_train, y_train)
y_pred1 = m.predict(X_test)
accuracy_score(y_test, y_pred1)
acc = accuracy_score(y_test, y_pred1) * 100








print(f"Scikit learn KNN classifier accuracy: {accuracy_score(y_test, y_pred1):.3}")
print("The accuracy for optimal k = {0} is {1}".format(optimal_n, acc))
