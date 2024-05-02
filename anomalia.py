from sklearn.svm import OneClassSVM
import numpy as np

class AnomalyClass:
    def __init__(self, label, id, kernel, nu):
        self.label = label  # Etiqueta de la clase
        self.id = id  # Identificador único de la clase
        self.z = []  # Lista para almacenar las muestras de la clase
        self.sample_labels = []  # Lista para almacenar las etiquetas de las muestras de la clase
        self.mean = None  # Media de las muestras de la clase
        self.sigma = None  # Desviación estándar de las muestras de la clase
        self.c = 1  # Parámetro de la clase
        self.omega = 1 # Parámetro de la clase

        self.svm_model = OneClassSVM(kernel=kernel, nu=nu)  # Modelo SVM para detección de anomalías

    def fit(self, data, data_label):
        self.z.append(data[0])  # Almacena la muestra
        self.sample_labels.append(data_label)  # Almacena la etiqueta de la muestra
        self.mean = np.mean(self.z, axis=0)  # Calcula la media de las muestras
        self.c += 1

        #print(np.array(data).reshape(-1, 1))   
        #self.svm_model.fit(np.array(data).reshape(-1, 1))  # Entrena el modelo SVM con las muestras proporcionadas
        self.svm_model.fit(data)  # Entrena el modelo SVM con las muestras proporcionadas

    def predict(self, data):
        #return self.svm_model.predict(np.array(data).reshape(-1, 1))  # Predice si los datos son anomalías
        return self.svm_model.predict(data)  # Predice si los datos son anomalías