
import pickle
from oupn import OUPN
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from ncm_cos import NearestMeanClassifier
import csv
import pandas as pd
import os

############################################# INPUT PARAMETERS ##############################################


#EXPERIMENT_NAME = sys.argv[1]             # Name of the experiment
#IDENTITY_TO_LEARN = int(sys.argv[2])      # Number of classes to learn
DATABASE = 'FaceCOX' #sys.argv[3]          # Name of the dataset 
INITIAL_KNOWN = 5 #int(sys.argv[4])       # Number of classes to learn initially
TRAINING_SEQUENCE=3
SUBSEQUENCE_NUMBER=11
EXPERIMENT_NAME='ncm_cos'
NUM_EXPERIMENTS = 10
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

def test_phase_labels(model:OUPN, test: list, index_correspondence: dict):
    # Test phase
    number_of_sequences_processed = 0
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

            y_true.append(true_label)
            y_pred.append(prediction_nosup)

    return y_true, y_pred

def calculate_confusion_matrix(model:OUPN, test: list, index_correspondence: dict, initial_known: int):
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
                if (prediction_nosup == -1):
                    y_true.append('DESCONOCIDOS')
                    y_pred.append('DESCONOCIDOS')
                else:
                    y_true.append('DESCONOCIDOS')
                    y_pred.append('INICIALES' if prediction_nosup < initial_known else 'APRENDIDOS')
            else:
                if (prediction_nosup == -1):
                    y_true.append('INICIALES' if true_label < initial_known else 'APRENDIDOS')
                    y_pred.append('DESCONOCIDOS')
                else:
                    y_true.append('INICIALES' if true_label < initial_known else 'APRENDIDOS')
                    y_pred.append('INICIALES' if prediction_nosup < initial_known else 'APRENDIDOS')

    # Calcula la matriz de confusión utilizando scikit-learn
    # confusion_matrix_3x3 = confusion_matrix(y_true, y_pred, labels=["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])

    # Imprime la matriz de confusión
    # print("Matriz de Confusión 3x3:")
    # print(confusion_matrix_3x3)

    return y_true, y_pred

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

    if not os.path.exists('experiments/'+EXPERIMENT_NAME):
        os.makedirs('experiments/'+EXPERIMENT_NAME)

    print("Loading dataset..."+DATABASE)
    with open('datasets/'+DATABASE+'_RN100_512D_full_splitted.obj', 'rb') as pickle_file:
        dataset = pickle.load(pickle_file)

    f1_score_results = []  # Lista para almacenar los resultados
    overlap_count_results = []  # Lista para almacenar los resultados

    ##### EXPERIMENTS #####

    for experiment in range(NUM_EXPERIMENTS):
        seed = int(str(experiment+1))
        random.Random(seed).shuffle(dataset)

        training_data = [i[0:TRAINING_SEQUENCE] for i in dataset]
        test_data = [i[-1] for i in dataset]
        evaluation_data = [i[TRAINING_SEQUENCE:-1] for i in dataset]
        #train_labels = [str(i) for i in range(len(training_data))]
        #test_labels = [str(i) for i in range(len(training_data))]

        # Inicializar las listas de etiquetas verdaderas y etiquetas predichas
        true_labels = []
        pred_labels = []

        index_correspondence = {}
        csv_data = {} # Acabar de implementar los csv
        for i in range(len(dataset)):
            '''
            if i < INITIAL_KNOWN:
                csv_data[str(i)] = [str(i)+'_0']
                index_correspondence[str(i)] = i
            else:
            '''
            index_correspondence[str(i)] = -1

        ############################################# PRE-INITIALIZATION ##############################################
        # Initialize the model with the ncm method to create 5 initial prototypes
        print("Initializing model...")
        #model.initialize(training_data)
        #user_counter=INITIAL_KNOWN

        nearest_mean_classifier = NearestMeanClassifier()

        # Crear una lista con los individuos
        individuos = list(range(len(training_data)))

        # Crear diccionario de indices
        #dic_etiquetas = {}

        user_counter = 0
        individuo = 0
        # Procesamos individuos hasta que los completamos todos o creamos los prototipos iniciales
        while individuos:
            # Elegimos un individuo al azar
            #individuo = random.Random(seed).choice(individuos)

            # Obtenemos la primera secuencia del individuo
            sequence = training_data[individuo][0]

            final_etiqueta = TRAINING_SEQUENCE - len(training_data[individuo])

            # Construimos la etiqueta del individuo
            etiqueta = str(individuo)+'_'+str(final_etiqueta)  # Ejemplo: 0_0

            # Procesamos la secuencia del individuo
            for frame in range(num_frames):
                predicted_label = nearest_mean_classifier.predict(sequence[frame], user_counter, etiqueta)

                if predicted_label == -1:
                    print("Creados todos los prototipos iniciales...")
                    break
                elif predicted_label == user_counter:
                    index_correspondence[str(predicted_label)] = individuo
                    user_counter += 1
                    csv_data[str(predicted_label)] = [etiqueta]
                elif predicted_label >= 0:
                    csv_data[str(predicted_label)].append(str(individuo)+'_'+str(final_etiqueta))
                #print(f"Csv_data[{str(predicted_label)}] = {csv_data[str(predicted_label)]}")
            
            
            else:
                # Eliminar la primera secuencia del individuo procesado del conjunto de entrenamiento
                training_data[individuo].pop(0)

                # Eliminar el individuo del procesamiento si ya no tiene más secuencias
                if not training_data[individuo]:
                    individuos.remove(individuo)
                    individuo += 1
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break

        ############################################# INITIALIZATION ##############################################
        # Creamos el modelo y le pasamos los prototipos iniciales
        model=OUPN(config)
        for index, prototype in enumerate(nearest_mean_classifier.prototypes):
            model.prototypes.append(prototype)
            print("Prototype ", index, " added to the model...")
        #print(model.prototypes)
        
        user_counter = len(model.prototypes)
        print("Model initialized with ", user_counter, " prototypes...")

        ############################################# EVALUATION ##############################################
        print("Starting evaluation process...")
        for step in range(SUBSEQUENCE_NUMBER):
            final = step + TRAINING_SEQUENCE - 1
            print(f"STEP {step+1} OF {SUBSEQUENCE_NUMBER}")
            
            print("Initial test phase...")
            accuracy, precision, recall, f1, size_unsup = test_phase(model, test_data, index_correspondence)

            for id_object, sequence in enumerate([i[step] for i in evaluation_data]):
                for frame in range(num_frames):
                    predicted_label=model.process_sequence(sequence[frame], user_counter, str(id_object)+'_'+str(final+1))
                    if predicted_label== user_counter:
                        index_correspondence[str(predicted_label)] = id_object
                        user_counter+=1
                        print ('NEW CLASS DETECTED: ', user_counter)
                        csv_data[str(predicted_label)] = [str(id_object)+'_'+str(final+1)]
                    elif predicted_label >= 0:
                        csv_data[str(predicted_label)].append(str(id_object)+'_'+str(final+1))
                    
                    if id_object < len(model.prototypes):
                        # Agregar las etiquetas verdaderas y predichas a las listas
                        true_labels.append(id_object)
                        pred_labels.append(predicted_label)
                    else:
                        # Agregar las etiquetas verdaderas y predichas a las listas
                        true_labels.append(-1)
                        pred_labels.append(predicted_label)

        # Final test phase
        accuracy, precision, recall, f1_score, size_unsup = test_phase(model, test_data, index_correspondence)
        # F1-Score
        f1_score_results.append({
            'Experimento': experiment + 1,  # Sumar 1 para empezar desde 1 en lugar de 0
            'Resultado': f1_score
        })
        print("F1-Score final: ", f1_score)
        # Overlap count
        overlap_count = model.measure_overlap(model.prototypes)
        overlap_count_results.append({
            'Experimento': experiment + 1,  # Sumar 1 para empezar desde 1 en lugar de 0
            'Resultado': overlap_count
        })
        print("Overlap count: ", overlap_count)

        # Create a list of tuples for each row
        data_eval = np.column_stack((true_labels, pred_labels))
        with open(f"experiments/{EXPERIMENT_NAME}/{seed}"+'_labels_eval.txt', 'a') as txtfile:
            np.savetxt(txtfile, data_eval, fmt='%d')

        y_true, y_pred = calculate_confusion_matrix(model, test_data, index_correspondence, INITIAL_KNOWN)
        # Create a list of tuples for each row
        data = np.column_stack((y_true, y_pred))
        # Append the data to the CSV file (or create a new file if it doesn't exist)
        with open(f"experiments/{EXPERIMENT_NAME}/"+'0_matriz_confusion_3_test.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(data)

        test_phase_true, test_phase_pred = test_phase_labels(model, test_data, index_correspondence)
        # Create a list of tuples for each row
        data_test = np.column_stack((test_phase_true, test_phase_pred))
        # Write the data to a text file
        with open(f"experiments/{EXPERIMENT_NAME}/"+'0_matriz_confusion_all_labels_test.txt', 'a') as txtfile:
            np.savetxt(txtfile, data_test, fmt='%d')


    ##### END EXPERIMENTS #####

    # Creamos los dataframe de los resultados
    df_f1 = pd.DataFrame(f1_score_results)
    df_overlap = pd.DataFrame(overlap_count_results)

    # Transponer el DataFrame
    df_f1_transpuesto = df_f1.T
    df_overlap_transpuesto = df_overlap.T

    # Escribir los DataFrame pivotados en un archivo Excel
    df_f1_transpuesto.to_csv(f'resultados_f1_score_{EXPERIMENT_NAME}.csv', index=False, header=False)
    df_overlap_transpuesto.to_csv(f'resultados_overlap_{EXPERIMENT_NAME}.csv', index=False, header=False) 

    '''
    confusion_matrix = calculate_confusion_matrix(model, test_data, index_correspondence)
    print("Matriz de Confusión 3x3:")
    #print(confusion_matrix)
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
    '''
    

