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
    
    for q in range(20): #Consideramos evaluar hasta un maximo de 20 vecinos
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
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training y 30% test

#Categorias -> Hay 11 features de este dataset
categoria = range(11)
feature = list()

#La idea es crear un conjunto de combinaciones para probar el rendimiento para distinta cantidad de features

rendimiento_optimo = 500
vecinos_optimo = 0
feature_optimo = list()

for i in range(10):  
    
    feature = Funciones.combinaciones(categoria,i+1)
    length = len(feature)
    
    #Para observar las combinaciones que se realizan
    #print(feature)
    #print("\n")
    
    for i in range(length):
    
        no_feature = np.delete(categoria,feature[i], axis = 0)
    
        #Me quedo con las feature/caracteristica que quiero analizar
        datos1 = np.delete(X_train, no_feature, axis = 0)
        datos2 = np.delete(X_test, no_feature, axis = 0)
        datos3 = np.delete(y_train, no_feature, axis = 0)
        datos4 = np.delete(y_test, no_feature, axis = 0)
        
        #Clasificacion KNN 
        precision_max , vecinos = KNN(datos1, datos3, datos2, datos4)
    
        if (rendimiento_optimo > precision_max):
            feature_optimo = feature[i]
            rendimiento_optimo = precision_max
            vecinos_optimo = vecinos

print("Configuracion de hiperparametros optimas")
print("Rendimiento maximo: ", rendimiento_optimo)
print("Para cantidad de vecinos k: ", vecinos_optimo)
print("Para las siguientes caracteristica/feature: ", feature_optimo)

#Bibliografia consultada
#https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
