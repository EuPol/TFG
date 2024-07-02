import numpy as np
import pickle
import random
import argparse
from loaders import load_data_to_process
from collections import OrderedDict
from Gaussian_Mixture_Model import Gaussian_Mixture_Model
from Scaling_Incremental_Learning import Scaling_Incremental_Learning
from Centroid_Based_Concept_Learning import Centroid_Based_Concept_Learning
from Nearest_Non_Outlier import Nearest_Non_Outlier
from statistics import mode
from finch import FINCH
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import gc


class Linear_Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(Linear_Classifier, self).__init__()
        self.fc = torch.nn.Linear(512, num_classes)#(2304, num_classes)

    def forward(self, x):
        return self.fc(x)


SoftMax = torch.nn.Softmax(dim=1)
threshold_softmax=0.8

# Args:
# -m: model to process
# -r: number of repetition

# create an ArgumentParser object
parser = argparse.ArgumentParser(description='Comparative methods')

# add positional arguments
parser.add_argument('model', type=str, help='Model using for evalluate the system')
parser.add_argument('rep', type=int, help='Number of repetition')
parser.add_argument('database', type=str, help='Number of repetition')

args = parser.parse_args()

if __name__=="__main__":
    torch.cuda.empty_cache()
    file_pi2 = open(args.database+'_RN100_512D_full_splitted.obj', 'rb') 
    data_set = pickle.load(file_pi2)
    random.shuffle(data_set)
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results={}
    index_correspondece={}

    # Initial train
    features_incremental=OrderedDict()
    user_labels=[i for i in range(5)]
    for i in user_labels:
        #features_incremental[i]=(torch.from_numpy(np.array(data_set[i][0])).to(device)).double()
        features_incremental[i]=(torch.from_numpy(np.array(data_set[i][0],dtype=np.float32))).double()#.to(device)
        index_correspondece[i]=i

    incremental_index=5
    remain_to_learn=80


    if args.model=='NNO':
        print('Using NNO')
        NCM=Nearest_Non_Outlier(tau=2.0, rank=4, learning_rate=0.001, batch_size=20, num_workers=1, max_number_of_epoch=200, gpu=0)
    elif args.model=='GMM':
        print('Using GMM')
        NCM=Gaussian_Mixture_Model()
    elif args.model=='CBL':
        print('Using CBL')
        NCM =Centroid_Based_Concept_Learning(3, 10)    
   
    #NCM=Nearest_Clasifier_Mean()
    #NCM=Gaussian_Mixture_Model()
    #NCM = Nearest_Non_Outlier()#Nearest_Clasifier_Mean()#Gaussian_Mixture_Model()#Centroid_Based_Concept_Learning(5, 0.7)#Scaling_Incremental_Learning(512)#Nearest_Clasifier_Mean()
    #NCM.fit(features_incremental)
    #NCM.fit(user_labels, features_incremental)

    #Linear Classifier
    model = Linear_Classifier(20)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    all_loss = []
    for epoch in range(10000):
        for index,user in enumerate(user_labels):
            X_train =features_incremental[user].type(torch.FloatTensor)
            y_train = torch.tensor([index]*len(X_train), dtype=torch.long)
            output = model(X_train)

            loss = criterion(output, y_train.view(-1))
            all_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
    #plt.plot(all_loss)
    #plt.show()
    # Añadir to device a partir de aquí
    # Incremental train

    #cluster_quality_checker = check_cluster_quality(svm_path)

    with torch.no_grad():
        number_of_discovered_classes = 0
        residual_dict = dict()
        clustered_dict = dict()
        probability_list = [0] * len(data_set)
        raw_list = [0] * len(data_set)
        image_list = []
        remain_to_add=324
    number_of_known_classes=20
    for step in range(1,12):
        print('STEP: '+str(step))
        for user in  tqdm(range(len(data_set)),  total=len(data_set)):#range(len(data_set)):
            features_unknown=(torch.from_numpy(np.array(data_set[user][step])))
            image_names=['image_'+str(user)+'_'+str(step)+'_'+str(i) for i in range(len(data_set[user][step]))]
            features_unknown=(torch.from_numpy(np.array(data_set[user][step])).to(device)).double()  # Comentar en caso de error
            Logit = model(features_unknown.type(torch.FloatTensor))
            softmax = SoftMax(Logit).double()
            features_unknown=features_unknown.double().to(device)
            

            sm, _ = torch.max(softmax, axis=1)
            predicted_known = sm >= threshold_softmax
            predicted_unknown = ~predicted_known
            is_unknown=mode(predicted_unknown.cpu().numpy())


            n = 1 + 20 + number_of_discovered_classes
            probability_tensor = torch.zeros(features_unknown.shape[0], n).double()
            probability_tensor[predicted_known, 1 : (1 + number_of_known_classes)] = softmax[predicted_known, :].cpu()

            FV = features_unknown.cpu()
            if torch.sum(predicted_unknown) > 0:
                if number_of_discovered_classes > 2:
                    FV_predicted_unknown = FV[predicted_unknown, :]
                    p_given_unknown = NCM.predict(FV_predicted_unknown.to(device)).cpu().double()
                    probability_tensor[predicted_unknown, 0] = p_given_unknown[:, 0]
                    probability_tensor[predicted_unknown, 1 + number_of_known_classes :] = p_given_unknown[:, 1:]
                else:
                    probability_tensor[predicted_unknown, 0] = 1.0
            
            normalized_tensor = probability_tensor
            probability_list[i] = normalized_tensor.detach().clone().cpu()
            image_list = image_list + list(image_names)
            nu = 0
            p_max, i_max = torch.max(normalized_tensor, axis=1)
            for k in range(normalized_tensor.shape[0]):
                if i_max[k] == 0:  # predicted unnkwon unknown
                    residual_dict[image_names[k]] = FV[k, :].numpy()
            #print(f"len(residual_dict) = {len(residual_dict)}")

            if len(residual_dict) >= 250 and remain_to_add>0:
                image_names_residual, FVs_residual = zip(*residual_dict.items())
                data = np.vstack(list(FVs_residual))
                c_all, num_clust, req_c = FINCH(data, verbose=False)
                if len(num_clust) <= 2:
                    index_partition_selected = 0
                else:
                    index_partition_selected = 1

                cluster_labels = c_all[:, index_partition_selected]
                number_of_clusters = num_clust[index_partition_selected]  # number of clusters after clustering.

                to_be_delete = []
                features_dict_incremental = OrderedDict()
                if number_of_clusters >= 4:#min_number_cluster_to_start_adaptation:
                    class_to_process = []
                    images_cluster = []
                    for cluster_number in range(number_of_clusters):  # number of clusters after clustering.
                        index = [iii for iii in range(len(cluster_labels)) if cluster_labels[iii] == cluster_number]
                        if len(index) >= 20:#min_number_point_to_create_class:
                            feature_this_cluster = torch.from_numpy(np.array([FVs_residual[jjj] for jjj in index]))
                            if 1 == 1:
                                
                                to_be_delete = to_be_delete + index
                                nu = nu + 1
                                class_number = int(nu + number_of_discovered_classes + number_of_known_classes)-1
                                features_dict_incremental[class_number] = torch.from_numpy(
                                    np.array([FVs_residual[jjj] for jjj in index])
                                )  # .double()#.cuda()
                                class_to_process.append(class_number)
                                images_cluster.append([int(image_names_residual[jjj].split('_')[1]) for jjj in index])

                    if ((len(class_to_process) > 0) and (number_of_discovered_classes > 0)) or (len(class_to_process) > 2):
                        NCM.fit(class_to_process, features_dict_incremental)
                        for iclass, real_class in enumerate(class_to_process):
                            index_correspondece[real_class]=mode(images_cluster[iclass])
                    else:
                        nu = 0
                        to_be_delete = []

                    if nu > 0:
                        image_covered = []
                        for j in to_be_delete:
                            image_covered.append(image_names_residual[j])
                        for name in image_covered:
                            fv_name = residual_dict[name]
                            clustered_dict.update({name: fv_name})
                            del residual_dict[name]
                    number_of_discovered_classes = number_of_discovered_classes + nu
                    remain_to_add-=nu
        #print(f"{nu} new classes added.")
        #print(f"len(clustered_dict) = {len(clustered_dict)}")
        #print(f"len(residual_dict) = {len(residual_dict)}")
        #print(f"Total discovered classes = {number_of_discovered_classes}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
           

           
            
        # Test phase
        results[step]=[]
        print ('INIT_TEST_PHASE: ',len(data_set))
        for user in range(len(data_set)):
            #test_feature=(torch.from_numpy(np.array(data_set[user][9])).to(device)).double()
            test_feature=(torch.from_numpy(np.array(data_set[user][13])))

            Logit = model(test_feature.type(torch.FloatTensor))
            softmax = SoftMax(Logit).double()
            test_feature=test_feature.double().to(device)

            sm, _ = torch.max(softmax, axis=1)
            predicted_known = sm >= threshold_softmax
            predicted_unknown = ~predicted_known
            is_unknown=mode(predicted_unknown.cpu().numpy())

            n = 1 + 20 + number_of_discovered_classes
            probability_tensor = torch.zeros(test_feature.shape[0], n).double()
            probability_tensor[predicted_known, 1 : (1 + number_of_known_classes)] = softmax[predicted_known, :].cpu()

            FV = test_feature.cpu()
            if torch.sum(predicted_unknown) > 0:
                if number_of_discovered_classes > 2:
                    FV_predicted_unknown = FV[predicted_unknown, :]
                    p_given_unknown = NCM.predict(FV_predicted_unknown.to(device)).cpu().double()
                    probability_tensor[predicted_unknown, 0] = p_given_unknown[:, 0]
                    probability_tensor[predicted_unknown, 1 + number_of_known_classes :] = p_given_unknown[:, 1:]
                else:
                    probability_tensor[predicted_unknown, 0] = 1.0
            
            normalized_tensor = probability_tensor
            probability_list[i] = normalized_tensor.detach().clone().cpu()
            p_max, i_max = torch.max(normalized_tensor, axis=1)

            

            for k in range(normalized_tensor.shape[0]):
                if i_max[k]==0:
                    predicted_label=-1
                else:
                    predicted_label=index_correspondece[int(i_max[k])-1]

                if user> (number_of_discovered_classes + number_of_known_classes):
                    real_user=-1
                else:
                    real_user=user
                results[step].append([real_user,predicted_label])
            

            #print("User: ", user, "Predicted label: ", predicted_label)

        TP=0
        FP=0
        FN=0
        TN=0
        for measure in results[step]:
            true_IOI,IOI_detected=measure
            if true_IOI < 0:
                if (IOI_detected == -1):
                    TN += 1.
                else:
                    FP += 1.
            else:
                if (IOI_detected == -1):
                    FN += 1.
                else:
                    if (IOI_detected == true_IOI):
                        TP += 1.
                    else:
                        FP += 1.

        # Calculate the metrics
        precision = float(TP) / float(max(TP + FP, 1))
        recall = float(TP) / float(max(TP + FN, 1))
        f1 = 2 * (precision * recall) / max(precision + recall, 0.001)
        accuracy = float(TP + TN) / float(max(TP + TN + FP + FN, 1))

        print("\t STEP COMPLETED")
        print("\t\tTrue positives: ", TP)
        print("\t\tFalse positives: ", FP)
        print("\t\tTrue negatives: ", TN)
        print("\t\tFalse negatives: ", FN)
        print("\n\n")
        print("┌──────────────┬──────────────┬──────────────┬──────────────┐")
        print("│   Accuracy   │   Precision  │    Recall    │   F1-Score   │")
        print("├──────────────┼──────────────┼──────────────┼──────────────┤")
        print("│    {:0.4f}    │    {:0.4f}    │    {:0.4f}    │    {:0.4f}    │".format(
            accuracy, precision, recall, f1))
        print("└──────────────┴──────────────┴──────────────┴──────────────┘")
        print("\n\n")
    
        print ('END_TEST_PHASE')   

    print ('EXPERIMENT_FINISHED')
    file_results = open(args.database+'/'+args.model+'/EXPERIMENT_RESULTS_'+args.model+'_'+str(args.rep)+'.obj', 'wb')
    pickle.dump(results, file_results)
