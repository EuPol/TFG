
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
