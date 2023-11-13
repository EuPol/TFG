
# Cluster structure
#   - z_full: samples clustered
#   - p: gaussian mean
#   - sigma: gaussian variance (isotropic)
#   - omega: gaussian weight  

import numpy as np
import sklearn.metrics.pairwise as sk


def distance(x, y):
    #calculate cosine distance
    return 1- np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_similarity(x, y):
    #calculate cosine similarity
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    #return sk.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]

def softmax(x):
    #calculate softmax
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    #calculate sigmoid
    return 1 / (1 + np.exp(-x))


class OUPN():

    def __init__(self, config):
        self.prototypes = []                         # K prototypes
        self.beta=config['beta']                     # β init
        self.gamma=config['gamma']                   # γ init
        self.tau=config['tau']                       # τ init 
        self.decay=config['decay']                   # Memory decay ρ
        self.aplha=config['alpha']                   # Threshold α
        self.lambda_entropy=config['lambda_entropy'] # Entropy loss λent (not used)
        self.lambda_cluster=config['lambda_cluster'] # New cluster loss λclu (not used)
        self.tau_ratio=config['tau_ratio']           # Pseudo label temperature ratio τ~ /τ 
        self.self_loss=0                             # Self loss  (not used)
        self.entropy_loss=0                          # Entropy loss (not used)
        self.prob_new=0                              # Probability of new cluster (not used)
        self.global_variance=0                       # Global variance
        self.LEARNED_IDS=config['id_to_learn']       # Ids to learn (0 if you want to learn in a open set scenario)


    def initialize(self, initial_data):
        sigma_global=[]
        global_mean=[]
        for index, sample in enumerate(initial_data):
            p_k=np.mean(sample, axis=0)
            global_mean.append(sample)
            sigma_k=np.mean(np.var(sample, axis=0))
            sigma_global.append(sigma_k)
            omega=1/len(initial_data)
            self.prototypes.append({"id":index,'z':sample,'sample_labels':[str(index)+'_0'], "p":p_k, "sigma":sigma_k, "c":10, "omega":1})  # c is an estimation of the samples number
        
        self.global_variance=1

        # Posiblle intialization with sigma depending parameters
        #self.beta=-2*0*self.global_variance
        #self.gamma=2*self.global_variance
        #self.tau=2*self.global_variance
    
    def create_prototype(self, sample, index, sample_label):
        self.prototypes.append({"id":index,'z':sample, 'sample_labels':[sample_label] ,"p":sample, "sigma":1,"c":1, "omega":1})


    def e_step(self,features, prototypes, beta, gamma, tau):
        y=[]
        for k in prototypes:
            y.append(np.log(k['omega'])-(1/tau)*distance(features, k['p']))
        y_smax=softmax(y)

        u= [sigmoid(elem) for elem in ((np.array([(1/tau)*distance(features, k['p']) for k in prototypes])-beta)/gamma)]
        return u,y
    
    def m_step(self,features, prototypes, u, y):
        y_label=np.argmax(y)
        for index,k in enumerate(prototypes):
            if index != y_label:
                k['omega']=self.decay*k['omega']
            else:
                k['p']=features/(k['c']*self.decay + 1) + (k['p']*self.decay*k['c'])/(k['c']*self.decay + 1)
                k['c']= k['c']*self.decay + 1
        
        #Suppose constant weights
    
    def process_sequence(self, sequence, new_label,sample_label):
        u,y=self.e_step(sequence, self.prototypes, self.beta, self.gamma, self.tau)
        
        if min(u)>self.aplha:
            if new_label<self.LEARNED_IDS:
                self.create_prototype(sequence, new_label,sample_label)
                return new_label
            else:
                return -1
        else:
            self.m_step(sequence, self.prototypes, u, y)
            self.prototypes[np.argmax(y)]['sample_labels'].append(sample_label)
            return np.argmax(y)
        
    def model_test(self,features):
        u,y=self.e_step(features, self.prototypes, self.beta, self.gamma, self.tau)
        return min(u),np.argmax(y)
    
    def bhattacharyya_distance(self, gaussian1, gaussian2):
        # Gaussian1 y Gaussian2 son diccionarios que contienen información sobre las gaussianas.
        # Por ejemplo, Gaussian1 = {"p": media1, "sigma": varianza1}, Gaussian2 = {"p": media2, "sigma": varianza2}

        mean1 = gaussian1["p"]
        var1 = gaussian1["sigma"]
        mean2 = gaussian2["p"]
        var2 = gaussian2["sigma"]

        # Calcular la distancia de Bhattacharyya
        term1 = 1/4 * np.log((var1/var2 + var2/var1 + 2) / 4)
        term2 = 1/4 * np.sum((mean1 - mean2)**2 / (var1 + var2), axis=-1)
        distance = term1 + term2

        return distance
    
    def measure_overlap_bhattacharyya(self, prototypes):
        overlap_count = 0
        total_pairs = 0

        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                # Calcula la distancia de Bhattacharyya entre las gaussianas i y j
                bhattacharyya_dist = self.bhattacharyya_distance(prototypes[i], prototypes[j])

                # Define un umbral para determinar si hay solapamiento
                threshold = 0.5  # Ajusta este valor según tus necesidades

                if bhattacharyya_dist <= -np.log(threshold):
                    overlap_count += 1

                total_pairs += 1

        overlap_percentage = (overlap_count / total_pairs) * 100.0
        return overlap_percentage
    
    def calculate_prototype_overlap(self):
        # Crea un diccionario donde las claves son etiquetas de identidad y los valores son listas de prototipos
        identity_prototype_dict = {}

        for i, ensemble in enumerate(self.prototypes):
            # Obtiene la etiqueta de identidad de la muestra utilizada para inicializar el prototipo
            identity_label = ensemble['sample_labels'][0].split('_')[0]
            
            if identity_label not in identity_prototype_dict:
                identity_prototype_dict[identity_label] = [i]
            else:
                identity_prototype_dict[identity_label].append(i)

        return identity_prototype_dict
    
    def measure_overlap(self, prototypes):
        
        identity_prototype_dict = self.calculate_prototype_overlap()
        #print(identity_prototype_dict)

        # Calcula el solapamiento de prototipos
        prototype_overlap_count = 0
        for label, prot in identity_prototype_dict.items():
            if len(prot) > 1:
                #prototype_overlap_count += len(prot)
                #prototype_overlap_count += 1
                prototype_overlap_count += len(prot) * (len(prot) - 1) / 2

        return prototype_overlap_count

    def calculate_prototype_identity_labels(self, output_file):
        # Calculate the identity labels of prototypes and save them to a text file.
        with open(output_file, 'w') as file:
            for i, ensemble in enumerate(self.prototypes):
                if ensemble['sample_labels']:
                    identity_label = ensemble['sample_labels'][0].split('_')[0]  # Assuming the first sample's label is the identity label
                    file.write(f'Prototipo {i}, {identity_label}\n')
                else:
                    file.write(f'Prototipo {i}, N/A\n')
    '''
    def calculate_prototype_overlap_2(self):
        # Crea un diccionario para rastrear qué prototipos se inicializaron con la misma entidad
        identity_prototype_dict = defaultdict(list)

        # Itera a través de los prototipos
        for idx, prototype in enumerate(self.prototypes):
            # Obtiene la entidad del prototipo actual
            entity = prototype['sample_labels'][0]

            # Agrega el índice del prototipo actual a la lista correspondiente a su entidad
            identity_prototype_dict[entity].append(idx)

        return identity_prototype_dict
    '''