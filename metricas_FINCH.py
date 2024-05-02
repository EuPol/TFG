import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

EXPERIMENT_NAME='finch_3_sec'

# Cargar las etiquetas verdaderas y predichas totales desde los archivos
directory = f"experiments/{EXPERIMENT_NAME}/"

etiquetas_verdaderas_totales = np.load(directory + 'etiquetas_verdaderas_totales.npy')
etiquetas_predichas_totales = np.load(directory + 'etiquetas_predichas_totales.npy')

'''
# Eliminar las etiquetas de un experimento
experimento_descartado = 3  # El cuarto experimento tiene índice 3
etiquetas_verdaderas_totales = np.delete(etiquetas_verdaderas_totales, experimento_descartado, axis=0)
etiquetas_predichas_totales = np.delete(etiquetas_predichas_totales, experimento_descartado, axis=0)
'''
# Calcular el porcentaje de acierto
acierto = accuracy_score(etiquetas_verdaderas_totales.flatten(), etiquetas_predichas_totales.flatten())
print("Porcentaje de acierto:", acierto)
print("\n")

# Inicializar el contador de solapamientos
contador_solapamientos = 0
total_solapamientos = 0

# Iterar sobre cada experimento
for i in range(len(etiquetas_verdaderas_totales)):
    contador_solapamientos = 0
    # Obtener las etiquetas verdaderas y predichas del experimento actual
    verdaderas = etiquetas_verdaderas_totales[i]
    predichas = etiquetas_predichas_totales[i]
    
    # Contar los solapamientos para el experimento actual
    for j in range(len(verdaderas)):
        if verdaderas[j] != predichas[j]:
            contador_solapamientos += 1
    
    print(f"Experimento {i + 1}: {contador_solapamientos} solapamientos")
    total_solapamientos += contador_solapamientos

# Calcular la cantidad total de solapamientos
print("\n")
print(f"Cantidad total de solapamientos: {total_solapamientos} de un total de {len(etiquetas_verdaderas_totales) * len(etiquetas_verdaderas_totales[0])} etiquetas")

'''
# Crear un histograma de las etiquetas verdaderas y predichas
bins = np.arange(-0.5, 5.5, 1)  # Define los bordes de las barras para cada etiqueta
plt.hist(etiquetas_verdaderas_totales.flatten(), bins=bins, alpha=0.5, label='Etiquetas Verdaderas', edgecolor='black')
plt.hist(etiquetas_predichas_totales.flatten(), bins=bins, alpha=0.5, label='Etiquetas Predichas', edgecolor='black')

plt.xlabel('Etiqueta')
plt.ylabel('Frecuencia')
plt.title('Histograma de Etiquetas')
plt.xticks(range(5))
plt.legend()
plt.grid(True)
plt.show()

'''
# Crear un histograma de las etiquetas verdaderas y predichas
# Definir las etiquetas posibles
etiquetas_posibles = np.arange(5)

# Contar la frecuencia de cada etiqueta
frecuencia_verdaderas = np.bincount(etiquetas_verdaderas_totales.flatten(), minlength=len(etiquetas_posibles))
frecuencia_predichas = np.bincount(etiquetas_predichas_totales.flatten(), minlength=len(etiquetas_posibles))

# Crear un histograma de las etiquetas verdaderas y predichas
plt.figure(figsize=(10, 5))

plt.bar(etiquetas_posibles - 0.2, frecuencia_verdaderas, width=0.4, label='Etiquetas Verdaderas')
plt.bar(etiquetas_posibles + 0.2, frecuencia_predichas, width=0.4, label='Etiquetas Predichas')

plt.xlabel('Etiqueta')
plt.ylabel('Frecuencia')
plt.title('Histograma de etiquetas obtenidas mediante FINCH')
plt.xticks(etiquetas_posibles)
plt.ylim(0, 400)
plt.legend()
plt.grid(True)
plt.savefig('Histograma_FINCH.pdf')  


