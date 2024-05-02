import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

EXPERIMENT_NAME = 'main02'
NUM_EXPERIMENTS = 10
initial_classes_to_test = range(1, 31)
max_true_label_overall = 0

def plot_confusion_matrix(matrix, labels):
    # Create an instance of ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

    # Configure the confusion matrix plot
    disp.plot(cmap=plt.cm.Blues, values_format="d")

    # Add title and labels
    plt.title('Confusion Matrix')

    # Rotate x-axis and y-axis labels for better readability
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(rotation=45, fontsize=7)

    # Adjust layout to make room for rotated labels
    plt.tight_layout()

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the plot to a file with a tight bounding box
    plt.savefig('confusion_matrix_3_eval_main02.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

if __name__ == '__main__':
    true_labels_all = []
    pred_labels_all = []

    # Find the overall maximum true label
    for initial_known in initial_classes_to_test:
        for experiment in range(NUM_EXPERIMENTS):
            seed = int(str(initial_known) + str(experiment + 1))
            file_path = f"experiments/{EXPERIMENT_NAME}/{seed}_labels_eval.txt"

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Split the line into true and pred values
                    true_val, pred_val = map(int, line.strip().split())

                    max_true_label_overall = max(max_true_label_overall, true_val)

    # Use the overall maximum true label as the number of clusters
    num_clusters = max_true_label_overall

    for initial_known in initial_classes_to_test:
        for experiment in range(NUM_EXPERIMENTS):
            seed = int(str(initial_known) + str(experiment + 1))
            file_path = f"experiments/{EXPERIMENT_NAME}/{seed}_labels_eval.txt"

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Split the line into true and pred values
                    true_val, pred_val = map(int, line.strip().split())

                    if pred_val == -1:
                        pred_labels_all.append("DESCONOCIDOS")
                    else:
                        pred_labels_all.append("INICIALES" if pred_val < initial_known else "APRENDIDOS")
                    if true_val == -1:
                        true_labels_all.append("DESCONOCIDOS")
                    else:
                        true_labels_all.append("INICIALES" if true_val < initial_known else "APRENDIDOS")

    # Calcular la matriz de confusión
    confusion_matrix_3x3 = confusion_matrix(true_labels_all, pred_labels_all, labels=["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])
    print("Matriz de Confusión 3x3:")
    print(confusion_matrix_3x3)
    plot_confusion_matrix(confusion_matrix_3x3, ["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])