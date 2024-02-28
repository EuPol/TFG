
import pickle
from oupn import OUPN
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

############################################# INPUT PARAMETERS ##############################################


#EXPERIMENT_NAME = sys.argv[1]             # Name of the experiment
#IDENTITY_TO_LEARN = int(sys.argv[2])      # Number of classes to learn
DATABASE = 'FaceCOX' #sys.argv[3]          # Name of the dataset 
INITIAL_KNOWN = 6 #int(sys.argv[4])       # Number of classes to learn initially
SUBSEQUENCE_NUMBER=13
EXPERIMENT_NAME='prueba'
num_run=0
num_frames=4

# Parameters

config = {
    'beta': 5.0,                          # β init
    'gamma': 1.0,                         # γ init
    'tau': 0.1,                           # τ init
    'decay': 0.995,                       # Memory decay ρ
    'alpha': 0.3,                         # Threshold α
    'lambda_entropy': 0.0,                # Entropy loss λent
    'lambda_cluster': 0.5,                # New cluster loss λclu
    'tau_ratio': 0.1,                     # Pseudo label temperature ratio τ~ /τ
    'id_to_learn': 80                     # Ids to learn (0 if you want to learn in a open set scenario)
}


############################################ OPERATION FUNCTIONS ############################################

def test_phase(model:OUPN, test: list, index_correspondence: dict):
    # Test phase
    number_of_sequences_processed = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for id_object, subsequence in enumerate(test):
        number_of_sequences_processed += 1
        for frame in range(num_frames):
            u,y = model.model_test(subsequence[frame])
            if u >model.aplha:
                prediction_nosup=-1
            else:
                prediction_nosup=y

            if prediction_nosup >= 0:
                prediction_nosup = index_correspondence[str(prediction_nosup)]

            if id_object < len(model.prototypes):
                true_label = id_object
            else:
                true_label = -1

            if true_label < 0:
                if (prediction_nosup == -1):
                    TN += 1.
                else:
                    FP += 1.
            else:
                if (prediction_nosup == -1):
                    FN += 1.
                else:
                    if (prediction_nosup == true_label):
                        TP += 1.
                    else:
                        FP += 1.

    # Calculate the metrics
    precision = float(TP) / float(max(TP + FP, 1))
    recall = float(TP) / float(max(TP + FN, 1))
    f1 = 2 * (precision * recall) / max(precision + recall, 0.001)
    accuracy = float(TP + TN) / float(max(TP + TN + FP + FN, 1))

    print("\t STEP COMPLETED")
    print("\t\tSequences proccesed: ", number_of_sequences_processed)
    print("\t\tTrue positives: ", TP)
    print("\t\tFalse positives: ", FP)
    print("\t\tTrue negatives: ", TN)
    print("\t\tFalse negatives: ", FN)
    size_unsup = len(model.prototypes)
    print("\t\t Gaussian Number: ", size_unsup)
    print("\n\n")
    print("┌──────────────┬──────────────┬──────────────┬──────────────┐")
    print("│   Accuracy   │   Precision  │    Recall    │   F1-Score   │")
    print("├──────────────┼──────────────┼──────────────┼──────────────┤")
    print("│    {:0.4f}    │    {:0.4f}    │    {:0.4f}    │    {:0.4f}    │".format(
        accuracy, precision, recall, f1))
    print("└──────────────┴──────────────┴──────────────┴──────────────┘")
    print("\n\n")

    return accuracy, precision, recall, f1, size_unsup

def calculate_confusion_matrix(model:OUPN, test: list, index_correspondence: dict):
    number_of_sequences_processed = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Inicializa las listas de etiquetas verdaderas y etiquetas predichas
    y_true = []
    y_pred = []

    for id_object, subsequence in enumerate(test):
        number_of_sequences_processed += 1
        for frame in range(num_frames):
            u,y = model.model_test(subsequence[frame])
            if u >model.aplha:
                prediction_nosup=-1
            else:
                prediction_nosup=y

            if prediction_nosup >= 0:
                prediction_nosup = index_correspondence[str(prediction_nosup)]

            if id_object < len(model.prototypes):
                true_label = id_object
            else:
                true_label = -1

            if true_label < 0:
                y_true.append("DESCONOCIDOS")
            else:
                y_true.append("INICIALES" if true_label < INITIAL_KNOWN else "APRENDIDOS")

            if prediction_nosup < 0:
                y_pred.append("DESCONOCIDOS")
            else:
                y_pred.append("INICIALES" if prediction_nosup < INITIAL_KNOWN else "APRENDIDOS")

    # Calcula la matriz de confusión utilizando scikit-learn
    confusion_matrix_3x3 = confusion_matrix(y_true, y_pred, labels=["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])

    # Imprime la matriz de confusión
    # print("Matriz de Confusión 3x3:")
    # print(confusion_matrix_3x3)

    return confusion_matrix_3x3

def plot_confusion_matrix(matrix, labels):
    # Crear una instancia de ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    # Configurar el gráfico de la matriz de confusión
    disp.plot(cmap=plt.cm.Blues, values_format="d")

    # Agregar título y etiquetas
    plt.title(f'Matriz de Confusión ({INITIAL_KNOWN} clases iniciales)')
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Verdaderas')

    # Guardar el gráfico en un archivo
    plt.savefig('Confusion_matrix.png')

    # Mostrar el gráfico
    plt.show()


############################################ MAIN ############################################

if __name__ == '__main__':

    print("Loading dataset..."+DATABASE)
    with open('datasets/'+DATABASE+'_RN100_512D_full_splitted.obj', 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)

    training_data = [i[0] for i in dataset[:INITIAL_KNOWN]]
    test_data = [i[-1] for i in dataset]
    evaluation_data = [i[1:-1] for i in dataset]
    train_labels = [str(i) for i in range(len(training_data))]
    test_labels = [str(i) for i in range(len(training_data))]


    index_correspondence = {}
    csv_data = {} # Acabar de implementar los csv
    for i in range(len(dataset)):
        if i < INITIAL_KNOWN:
            csv_data[str(i)] = [str(i)+'_0']
            index_correspondence[str(i)] = i
        else:
            index_correspondence[str(i)] = -1

    print("Initializing model...")
    model=OUPN(config)
    model.initialize(training_data)
    user_counter=INITIAL_KNOWN

    print("Starting evaluation process...")
    for step in range(SUBSEQUENCE_NUMBER):
        print(f"STEP {step+1} OF {SUBSEQUENCE_NUMBER}")
        
        print("Initial test phase...")
        accuracy, precision, recall, f1, size_unsup = test_phase(model, test_data, index_correspondence)

        for id_object, sequence in enumerate([i[step] for i in evaluation_data]):
            for frame in range(num_frames):
                predicted_label=model.process_sequence(sequence[frame], user_counter, str(id_object)+'_'+str(step))
                if predicted_label== user_counter:
                    index_correspondence[str(predicted_label)] = id_object
                    user_counter+=1
                    print ('NEW CLASS DETECTED: ', user_counter)
                    csv_data[str(predicted_label)] = [str(id_object)+'_'+str(step)]
                elif predicted_label >= 0:
                    csv_data[str(predicted_label)].append(str(id_object)+'_'+str(step+1))

    overlap_count = model.measure_overlap(model.prototypes)
    print("Overlap count: ", overlap_count)
    #output_file = 'prototype_identity_labels.txt'
    #model.calculate_prototype_identity_labels(output_file)

    confusion_matrix = calculate_confusion_matrix(model, test_data, index_correspondence)
    print("Matriz de Confusión 3x3:")
    print(confusion_matrix)
    plot_confusion_matrix(confusion_matrix, ["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])

    csv_final = {}  # Acabar de implementar los csv
    for index, ensemble in enumerate(model.prototypes):
        csv_final[str(index)] = []
        for classifier in ensemble['sample_labels']:
            csv_final[str(index)].append(classifier)

    cluster_metrics = {} # Acabar de implementar los csv
    for user in csv_final:
        cluster_metrics[user] = {}
        pred_labels = [int(sequence_id.split('_')[0])
                       for sequence_id in csv_final[user]]
        true_labels = [int(csv_final[user][0].split('_')[0])
                       for sequence_id in csv_final[user]]
        cluster_metrics[user]['pred_labels'] = pred_labels
        cluster_metrics[user]['true_labels'] = true_labels

    print("Saving results...")
    with open(f"experiments/{EXPERIMENT_NAME}/{str(num_run)}"+'_classifier_final.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_metrics, f, ensure_ascii=False, indent=4)

    

