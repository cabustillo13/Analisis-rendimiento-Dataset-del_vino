from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

import Funciones

##############################################################
###########           CLASIFICADOR KNN             ###########
##############################################################

def KNN(X_train, y_train, X_test, y_test):
    
    precision_max = 500 #Valor semilla, sabemos que el porcentaje que obtendremos siempre sera menor o igual a 100
    vecinos = 0         # Cantidad de vecinos con los que se obtuvo el maximo rendimiento
    
    for q in range(20):
        #Crear el clasificador KNN
        knn = KNeighborsClassifier(n_neighbors=(q+1)) #El for arranca desde 0

        #Entrenar el modelo utilizando los training dataset
        knn.fit(X_train, y_train)

        #Prediccion para el dataset de test
        y_pred = knn.predict(X_test)

        #Precision del modelo en porcentaje, osea que tanto predice correctamente
        precision = 100*(metrics.accuracy_score(y_test, y_pred))
        
        if (precision_max > precision):
            precision_max = precision
            vecinos = (q+1)
    
    return (precision_max , vecinos)

##############################################################
###########                 MAIN                    ###########
##############################################################

#Cargar dataset
wine = datasets.load_wine()

#Dividir dataset para el train y test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test

#Categorias -> Hay 11 features de este dataset
categoria = range(11)
feature = list()

#La idea es crear un conjunto de combinaciones para probar el rendimiento para distinta cantidad de features
for i in range(10): 
    
    feature = Funciones.combinaciones(categoria,i+1)
    length = len(feature)
    
    no_feature = np.delete(categoria,feature[i], axis = 0)
    

#print(feature)
    #feature = np.delete(categoria, feature, axis=1) #Solo se deja los elementos de la lista que no vamos a utilizar para luego eliminarlas de la lista de test y train 

##ME QUEDE ACA
#y = np.array([[1,2,3,9],[4,5,6,9],[7,8,9,9],[10,11,12,9]])
#a=np.delete(y, [0,1], axis=1)
#print(a)



##COMENTE DESDE AQUI
##abcisas = list()
##ordenadas = list()

##for q in range(20):

    #Create KNN Classifier
##    knn = KNeighborsClassifier(n_neighbors=(q+1)) #El for arranca desde 0

    #Train the model using the training sets
##    knn.fit(X_train, y_train)

    #Predict the response for test dataset
##    y_pred = knn.predict(X_test)

#print(y_pred)
#print(y_pred[0])
#print(y_pred[1])
#print(y_pred[7])
#print(y_pred[8])

    # Model Accuracy, how often is the classifier correct?
##    precision = metrics.accuracy_score(y_test, y_pred)
##    print("Accuracy:", precision)
    
##    abcisas.append(q+1)
##    ordenadas.append(precision)

#print(wine.feature_names)

#plt.plot(abcisas , ordenadas)
#plt.xlabel('# de vecinos k')
#plt.ylabel('Precision')
#plt.title('Precision vrs cantidad de vecinos k')
#plt.show()

