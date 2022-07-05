#from msvcrt import LK_LOCK
from matplotlib.ft2font import HORIZONTAL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets #para mapas de color
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs,make_circles,make_classification, make_moons
#clasificadores
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas
#instanciar los clasificadore
classifiers = {
            'KNN':KNeighborsClassifier(3), #k vicinos mas cercanos se va a probar con 3
            'SVM':SVC(gamma=2,C=1),#iperparametros gamma y C
            'GP':GaussianProcessClassifier(1.0 * RBF(1.0)), #tecnicas usadas en cosas de audios RBF funcion que se importo
            'DT':DecisionTreeClassifier(max_depth=3), #max_depth es el numero de ramas que se quiere que tenga el arbol
            'MLP':MLPClassifier(alpha=1,max_iter=1000),#por defecto una capa de 100 neuronas con iteraciones de1000
            'NB':GaussianNB() #basada en el teorma de Bayes
    }
#datos de prueba con 3 data set distintos
#se utilizara la funcion  make_classification
# parametros 2 caracteristicas es decir en dos variables , no nos importan que sean redundates por eso n_redundantes=0  es decir que no nos interesa que se repitan
# que sean infromativas si, una variable no informativa es  que no aporte informacion al problema
# un clusters por cada clase por eso uno en n_clusters_per_class=1 
x,y = make_classification(n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1)
#al final make_classification nos devuelve 2 clousters mas o menos linealmente separables

#crear una variable aleatoria
rng = np.random.RandomState(2) #The instruction rng = np.random.RandomState(2) creates a random number generator with a seed of 2.
x+= 1* rng.uniform(size=x.shape) #con esto agragamos ruido a x para que no sea un problema de clasificacion tan sencillo
#b=1*rng.uniform(size=x.shape)
#print(rng.uniform(size=(5,2)))  5 vectores de 2 dimensiones con valores aleatorios de 0 a 1
linearly_separable = (x,y) #insertamos nuestras variables x y y la x ya con ruido incluido en una tupla
print("vector x")
print(type(x))
print("vector y")
print(type(y))
#delete the last 5 elements of x

#los otros 2 dataset van a ser make_moons y make_circles con algo de ruido ambos 
datasets = [make_moons(noise=0.1),make_circles(noise=0.1,factor=0.5),linearly_separable]


model_name= 'KNN'

#mapas de color
cm = plt.cm.RdBu #colormap pintar rojo y azul
cm_bright = ListedColormap(['#FF0000', '#0000FF']) #crear mapas de color para que no agarre los colores de default

figure = plt.figure(figsize=(9,3))
h=0.02 #step
i=1 #counter

for ds_cnt,ds in enumerate(datasets):
    x,y = ds
    x= StandardScaler().fit_transform(x) #normalizar los datos
    x_train,x_test,y_train,y_test = train_test_split(x,y) #separar los datos en train y test y se hace para probar si se esta sobreentrenando 
    if ds_cnt == 2:
        train = pandas.read_csv('train_0.csv')
        test = pandas.read_csv('test_0.csv')
        x_train = train.iloc[:,0:2] 
        y_train = train.iloc[:,2]
        x_test = test.iloc[:,0:2]
        y_test = test.iloc[:,2] 
        #corvert to numpy arrays 
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
    #lengh of the x_train
    print("lenght of x_train",(x_train))
    print("lenght of x_test",(y_test))
    model = classifiers[model_name] #instanciar el clasificador
    model.fit(x_train,y_train) #entrenar el clasificador
    score_train = model.score(x_train,y_train) #calcular el score del clasificador
    score_test = model.score(x_test,y_test) #calcular el score del clasificador
    print(score_train,score_test)
    #Graficar
    #como voy a tener 3 graficas una por cada dataset se necesita la funcion subplot que me permite meter distintas graficas en una sola fiugra de canvas
    ax=plt.subplot(1,3,i)
    #sacar los limites de donde tengo que graficar 
    x_min,x_max=x[:,0].min()-0.5,x[:,0].max()+0.5 #con el 0.5 un small desfase
    y_min,y_max=x[:,1].min()-0.5,x[:,1].max()+0.5
    #para poder graficar lo que hace el clasificador necesito crear una regilla con mesgrid
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h)) #con paso de h=0.02
    #esto creara una rejilla de puntos y cada punto lo voy a evaluar y con lo que me de la evaluacion voy a dibujar la curva
    #del clasificador, la curva que se aprende
    #para graficar un modelo no esta tan estadarizado por lo que se puede hacer de 2 formas
    if hasattr(model,"decision_function"): #la funcion pregunta si tiene cierto atributo
        #predecir los puntos de la rejilla de la siguiente manera
        zz=model.decision_function(np.c_[xx.ravel(),yy.ravel()]) #con esto se calcula la decision de cada punto
        print(zz)
    else:
        zz = model.predict_proba(np.c_[xx.ravel(),yy.ravel()]) #con esto se calcula la decision de cada punto

    #zz=zz.reshape(xx.shape) #a la salida de doy un reshape para que tenga forma de rejilla
    #ax.contourf(xx,yy,zz,cmap=cm,alpha=0.8) #con esto se dibuja la curva, alpha es el transparencia,cm es el mapa de color y xx,yy son los puntos y zz es la decision
    #puntos de entrenamiento
    ax.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_bright,edgecolors='k') #con esto se dibuja los puntos de train,c es el color,cmap es el mapa de color,edgecolors es el color de los bordes en este caso es black
    #puntos de prueba
    ax.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_bright,marker='x',edgecolors='g',alpha=0.6) #con esto se dibuja los puntos de test
    ax.set_xlim(x_min,x_max) #con esto se limita el eje x
    ax.set_ylim(y_min,y_max) #con esto se limita el eje y
    ax.set_xticks(()) #con esto se oculta los ticks en el eje x
    ax.set_yticks(()) #con esto se oculta los ticks en el eje y

    ax.text(0.05,0.95,('%.2f' % score_train).lstrip('0'),transform=ax.transAxes,size=15,ha='left',va='top') #con esto se pone el score en el eje x
    ax.text(0.05,0.05,('%.2f' % score_test).lstrip('0'),transform=ax.transAxes,size=15,ha='left',va='bottom') #con esto se pone el score en el eje x
    
    
    #esto se hace para poder ver si se estan sobreentrenando o no

    i=i+1

plt.tight_layout() #con esto se ajusta el layout de las graficas, para que no se superpongan, las junta para que no salgan tan retiradas
plt.show()