from functools import partial
from tkinter import *
from tkinter import Tk, Label
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import mysql.connector


mydb = mysql.connector.connect(
    host="localhost",
    user="missi",
    passwd="missi"
)


def insert_iris(sepal_length, sepal_width, petal_length, petal_width, classe):
    sql = " INSERT INTO data_iris (sepal_length ,sepal_width ,petal_length ,petal_width ,classe) VALUES (%s,%s,%s,%s,\"%s\")"
    val = (sepal_length, sepal_width, petal_length, petal_width, classe)
    mycursor.execute(sql, val)
    mydb.commit()


def insert_list_iris(iris):
    for i, t in zip(iris.data, iris.target):
        insert_iris(i[0], i[1], i[2], i[3], iris.target_names[t])


def read_iris():
    mycursor.execute("SELECT * FROM data_iris")
    return mycursor.fetchall()



mycursor = mydb.cursor()

# creation de la base de donnés mydatabase
mycursor.execute("CREATE DATABASE IF NOT EXISTS mydatabase")

# Utilisation de la base
mycursor.execute("USE mydatabase")

# création d'une table data_iris dans ma base de donnés
mycursor.execute(
    "CREATE TABLE IF NOT EXISTS data_iris (id INT AUTO_INCREMENT PRIMARY KEY, sepal_length FLOAT,sepal_width FLOAT,petal_length FLOAT,petal_width FLOAT,classe VARCHAR(255))")

# chargement des données iris
iris = datasets.load_iris()

# insertions
insert_list_iris(iris)

# Lecture
result = read_iris()



def regression(result):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # importation des données
    X = result[:,:2] # si ça marche pas
    #X = result #essayer avec cette syntax, car ça marche pas postgres pour l'instant pour moi
    Y = iris.target
    lab = iris.target_names

    # Division de la BD
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Création d'une instance et fiter les données
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Présentation des données
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

button = Button(root, text='Clustring', command=partial(regression,result),
                height = 3, width = 20,bg='#FFF8DC', fg='black')

button.grid(column=0, row=1,sticky= N)

root.mainloop() # Lancement de la boucle principale