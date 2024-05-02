from sklearn.covariance import MinCovDet
import numpy as np

class AnomalyClass:
    def __init__(self, label, id):
        self.label = label  
        self.id = id  
        self.z = []  
        self.sample_labels = []  
        self.mean = None  
        self.sigma = None  
        self.c = 1  
        self.omega = 1 
        
        self.mcd_model = MinCovDet()  

    def fit(self, data, data_label):
        self.z.append(data[0])  
        self.sample_labels.append(data_label)  
        self.mean = np.mean(self.z, axis=0)  
        self.c += 1

        self.mcd_model.fit(data)  

    def predict(self, data, threshold=2.0):
        mahalanobis_distances = self.mcd_model.mahalanobis(data)
        is_inlier = mahalanobis_distances < threshold
        return np.where(is_inlier, 1, -1)
