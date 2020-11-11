import mysql.connector
from sklearn import datasets

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
print(result)


