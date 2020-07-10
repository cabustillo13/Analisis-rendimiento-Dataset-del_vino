# Analisis-rendimiento-Dataset-del_vino
Obtener el mejor rendimiento para el dataset de vinos de sklearn utilizando un algoritmo KNN.

Dataset obtenido de sklearn: load.wine

Se presentan 11 características de entrada (obtenidad a partir de pruebas fisicoquímicas):  
'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines' y 'proline'.

Y a la salida hay tres tipos de clases, haciendo referencia a los 3 tipos de cultivares en Italia, donde se tomaron los datos.

Los parámetros que se evaluaron fueron: cantidad de vecinos k y combinación de características.

Debido a que los datos no están normalizados y el desequilibrio en la cantidad de datos para cada clase, cada vez que se ejecuta el programa se puede llegar a obtener distintas soluciones.

La ventaja de utilizar KNN radica en que es más rápido en comparación a algunos algortimos de clasificación. Las desventajas son el elevado costo y tiempo en la fase test,no es adecuado para los datos de grandes dimensiones y las características con altas magnitudes pesarán más que las características con bajas magnitudes.
