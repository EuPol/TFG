import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

EXPERIMENT_NAME = 'main_10'

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
    plt.savefig('confusion_matrix_3_test_main_10.pdf', bbox_inches='tight')

    # Show the plot
    plt.show()




if __name__ == '__main__':
    # Load the data from the CSV file
    with open(f"experiments/{EXPERIMENT_NAME}/"+'0_matriz_confusion_3_test.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)  # Read the headers
        loaded_data = np.array(list(csv_reader))

    # Extract true and predicted labels from the loaded data
    true_labels_loaded = loaded_data[:, 0]
    pred_labels_loaded = loaded_data[:, 1]

    # Calcular la matriz de confusión
    confusion_matrix_3x3 = confusion_matrix(true_labels_loaded, pred_labels_loaded, labels=["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])
    print("Matriz de Confusión 3x3:")
    print(confusion_matrix_3x3)
    plot_confusion_matrix(confusion_matrix_3x3, ["INICIALES", "APRENDIDOS", "DESCONOCIDOS"])