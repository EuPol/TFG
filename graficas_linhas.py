import matplotlib.pyplot as plt
import pandas as pd
import sys

name = str(sys.argv[2])
tipo_grafica = str(sys.argv[1])

if tipo_grafica == 'f1':
    # Cargar el archivo CSV
    df = pd.read_csv(f'resultados_f1_score_{name}.csv')

    # Obtener las columnas del DataFrame
    columnas = df.columns

    fig,ax = plt.subplots(figsize=(8,6))
    plt.rcParams['legend.title_fontsize'] = 18

    ax.set_title('F1-Score modelo base',fontsize=16)
    ax.set_xlabel('Experimento',fontsize=14)
    ax.set_ylabel('F1-Score',fontsize=14)

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    # Usar números enteros en el eje x del 0 al 1
    plt.xticks(range(1, len(columnas)+1), range(1, len(columnas)+1), rotation=45, ha='right')
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_fontsize(10)

    # Muestra los valores en el eje y
    ax.plot(range(1, len(columnas)+1), df.values.flatten(), linestyle='solid', marker='o', label='F1-Score')

    # Añadir recuadro con estadísticas
    max_value = df.max().max()
    min_value = df.min().min()
    mean_value = df.mean(axis=1).mean()
    std_value = df.std(axis=1).mean()

    stats_text = f'Máx: {max_value:.3f}\nMin: {min_value:.3f}\nMedia: {mean_value:.3f}\nDesv. Típica: {std_value:.3f}'

    ax.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom',
                bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))

    # Ajustar el rango del eje y entre 0 y 1
    plt.ylim(0, 1)

    # Mostrar el gráfico
    plt.savefig(f'F1-Score_grafico_{name}.pdf')  

elif tipo_grafica == 'overlap':
    # Cargar el archivo CSV
    df = pd.read_csv(f'resultados_overlap_{name}.csv')

    # Obtener las columnas del DataFrame
    columnas = df.columns

    fig,ax = plt.subplots(figsize=(8,6))
    plt.rcParams['legend.title_fontsize'] = 18

    ax.set_title('Cantidade de solapamentos modelo base',fontsize=16)
    ax.set_xlabel('Experimento',fontsize=14)
    ax.set_ylabel('Número de solapamentos',fontsize=14)

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    # Usar números enteros en el eje x del 0 al 1
    plt.xticks(range(1, len(columnas)+1), range(1, len(columnas)+1), rotation=45, ha='right')
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_fontsize(10)

    # Muestra los valores en el eje y
    ax.plot(range(1, len(columnas)+1), df.values.flatten(), linestyle='solid', marker='o', label='F1-Score')

    # Añadir recuadro con estadísticas
    max_value = df.max().max()
    min_value = df.min().min()
    mean_value = df.mean(axis=1).mean()
    std_value = df.std(axis=1).mean()

    stats_text = f'Máx: {max_value:.3f}\nMin: {min_value:.3f}\nMedia: {mean_value:.3f}\nDesv. Típica: {std_value:.3f}'

    ax.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', alpha=0.1, facecolor='white'))

    # Ajustar el rango del eje y entre 0 y 1
    plt.ylim(0, 20)

    # Mostrar el gráfico 
    plt.savefig(f'Overlap_grafico_{name}.pdf')  

else:
    print('Tipo de gráfica no reconocido')
    sys.exit(1)
