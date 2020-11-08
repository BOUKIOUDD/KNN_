import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# importation des données
iris = datasets.load_iris()
X = iris.data[:, :2] 
Y = iris.target
lab=iris.target_names

# Division de la BD
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
                                                    
# Création d'une instance et fiter les données
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Présentation des données
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])


from matplotlib.colors import ListedColormap
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(('m', 'c', 'y')))


plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=ListedColormap(('m', 'c', 'y')))
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.axis('equal')
plt.scatter(X[lab == 0], Y[lab == 0], color='m', label='setosa')
plt.scatter(X[lab == 1], Y[lab == 1], color='c', label='versicolor')
plt.scatter(X[lab == 2], Y[lab == 2], color='y', label='virginica')
plt.legend()
plt.title('Classification BD Iris par KNN')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()