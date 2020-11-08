from functools import partial
from tkinter import *
from tkinter import Tk, Label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def regression():
    # import some data
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    Y = iris.target

    # Split Dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # # Create an instance of Logistic Regression Classifier and fit the data
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # Prédictions
    y_pred = classifier.predict(X_test)

    # Matrice de confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z[1])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('IRIS PLAN')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

#root = Tk() # Création de la fenêtre racine


root = Tk(className='Clustring App')
# set window size
#root.geometry("400x400")

#set window color
root.configure(bg='blue')

lab0 = Label(root, text='welcome to the iris clustring !',foreground="#892222",
              background="#00BFFF", padx="50", pady="30")

lab1 = Label(root, text='All right reserved',foreground="#892222",
              background="#00FF7F", padx="83", pady="10")

lab0.grid(column=0, row=0)
lab1.grid(column=0, row=2,sticky= W)
label = Label(root, text='')

button = Button(root, text='Clustring', command=partial(regression),
                height = 3, width = 20,bg='#FFF8DC', fg='black')

button.grid(column=0, row=1,sticky= N)

root.mainloop() # Lancement de la boucle principale