
import numpy as np

def mahalanobis_distance(x, mean, covariance):
    # calculate mahalanobis distance
    diff = x - mean
    inv_covariance = np.linalg.inv(covariance)
    try:
        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_covariance), diff.T))
    except np.linalg.LinAlgError:
        # Manejar el caso cuando la matriz de covarianza no es invertible
        mahalanobis_dist = np.nan
    return mahalanobis_dist

def distance(x, y):
    #calculate cosine distance
    return 1- np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def distance_log(x, y):
    # Calculate logarithmic distance
    log_distance = np.sqrt(np.sum((np.log1p(x) - np.log1p(y))**2))
    return log_distance

def distance_euclidean(x, y):
    # Calculate Euclidean distance
    euclidean_distance = np.sqrt(np.sum((x - y)**2))
    return euclidean_distance

class NearestMeanClassifier:

    def __init__(self):
        self.prototypes = []
        self.threshold = 0.7
        #self.threshold = 0
        self.initial_prototypes = 5

    def predict(self, sequence, index, sample_label):
        if not self.prototypes:
            # Si no hay prototipos, crea uno con la primera secuencia
            self.prototypes.append({"id": index, 'z': [sequence], 'sample_labels': [sample_label],
                                    #"p": np.mean(sequence, axis=0),
                                    "p": sequence,
                                    "sigma": 1, "c": 1, "omega": 1})
            return 0  # Devuelve el índice del prototipo creado
        else:
            if len(self.prototypes) < self.initial_prototypes:
                # Calcula la distancia a cada prototipo
                #distances = [mahalanobis_distance(np.mean(sequence, axis=0), prototype['p'], prototype['covariance']) for prototype in self.prototypes]
                #distances = [dist for prototype in self.prototypes for dist in distance(sequence, prototype['p'])]
                distances = [np.mean(distance_euclidean(sequence, prototype['p'])) for prototype in self.prototypes]
                nearest_prototype_index = np.argmin(distances)

                #print(distances)
                #print(nearest_prototype_index)
                '''
                if len(self.prototypes) == 4 and distances[nearest_prototype_index] > self.threshold:
                    with open('experiments/ncm_eu/distances.txt', 'a') as txtfile:
                        np.savetxt(txtfile, distances)
                '''
                        
                # Compara con un umbral para decidir si asignar a un prototipo existente o crear uno nuevo
                if distances[nearest_prototype_index] < self.threshold:
                    #print("Distancia: ", distances[nearest_prototype_index])
                    # Asigna a un prototipo existente
                    nearest_prototype = self.prototypes[nearest_prototype_index]
                    nearest_prototype['z'] = np.concatenate((nearest_prototype['z'], [sequence]))
                    nearest_prototype['p'] = np.mean(nearest_prototype['z'], axis=0)
                    #nearest_prototype['covariance'] = np.cov(nearest_prototype['z'], rowvar=False)
                    nearest_prototype['c'] += 1
                    nearest_prototype['sample_labels'].append(sample_label)
                    return nearest_prototype_index
                else:
                    # Crea un nuevo prototipo
                    self.prototypes.append({"id": index, 'z': [sequence], 'sample_labels': [sample_label],
                                            #"p": np.mean(sequence, axis=0),
                                            "p": sequence,
                                            "sigma": 1, "c": 1, "omega": 1})
                    return len(self.prototypes) - 1  # Devuelve el índice del prototipo creado
            else:
                # si ya hay 5 prototipos, termina y devuelve -1
                return -1
        '''
        def predict(self, sequence, index, sample_label):
        if not self.prototypes:
            # Si no hay prototipos, crea uno con la primera secuencia
            self.prototypes.append({"id":index,'z':sequence, 'sample_labels':[sample_label] ,"p":sequence, "sigma":1,"c":1, "omega":1})

            return 0  # Devuelve el índice del prototipo creado
        else:
            if len(self.prototypes) < self.initial_prototypes:
                # Calcula la distancia a cada prototipo
                #distances = [mahalanobis_distance(np.mean(sequence, axis=0), prototype['p'], prototype['covariance']) for prototype in self.prototypes]
                #distances = [dist for prototype in self.prototypes for dist in distance(sequence, prototype['p'])]
                distances = [np.mean(distance(sequence, prototype['p'])) for prototype in self.prototypes]
                nearest_prototype_index = np.argmin(distances)

                print(distances)
                #print(nearest_prototype_index)

                # Compara con un umbral para decidir si asignar a un prototipo existente o crear uno nuevo
                if distances[nearest_prototype_index] < self.threshold:
                    #print("Distancia: ", distances[nearest_prototype_index])
                    # Asigna a un prototipo existente
                    nearest_prototype = self.prototypes[nearest_prototype_index]
                    nearest_prototype['z'] = np.concatenate((nearest_prototype['z'], sequence))
                    nearest_prototype['p'] = np.mean(nearest_prototype['z'], axis=0)
                    #nearest_prototype['covariance'] = np.cov(nearest_prototype['z'], rowvar=False)
                    nearest_prototype['c'] += 1
                    nearest_prototype['sample_labels'].append(sample_label)
                    return nearest_prototype_index
                else:
                    # Crea un nuevo prototipo
                    self.prototypes.append({"id":index,'z':sequence, 'sample_labels':[sample_label] ,"p":sequence, "sigma":1,"c":1, "omega":1})
                    return len(self.prototypes) - 1  # Devuelve el índice del prototipo creado
            else:
                # si ya hay 6 prototipos, termina y devuelve -1
                return -1
        '''



