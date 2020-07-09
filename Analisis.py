#Import scikit-learn dataset library
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

#Load dataset
wine = datasets.load_wine()

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3) # 70% training and 30% test

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

abcisas = list()
ordenadas = list()

for q in range(20):

    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=(q+1)) #El for arranca desde 0

    #Train the model using the training sets
    knn.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = knn.predict(X_test)

#print(y_pred)
#print(y_pred[0])
#print(y_pred[1])
#print(y_pred[7])
#print(y_pred[8])

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    precision = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", precision)
    
    abcisas.append(q+1)
    ordenadas.append(precision)
    
plt.plot(abcisas , ordenadas)
plt.xlabel('# de vecinos k')
plt.ylabel('Precision')
plt.title('Precision vrs cantidad de vecinos k')
plt.show()

