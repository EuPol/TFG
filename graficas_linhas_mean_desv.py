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
    df['threshold'] = [i / 10.0 for i in range(1, 13)]

    fig,ax = plt.subplots(figsize=(8,6))
    plt.rcParams['legend.title_fontsize'] = 18

    ax.set_title('F1-Score umbral euclídea',fontsize=16)
    ax.set_xlabel('Umbral',fontsize=14)
    ax.set_ylabel('F1-Score',fontsize=14)

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    plt.xticks(rotation = 45)
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        
    # Calculas la media y la desviacion típica de las columnas
    df['average'] = df[columnas].mean(axis=1)
    df['std'] = df[columnas].std(axis=1)

    # Se representa los valores medios y la desviación del eje X
    ax.plot(df['threshold'], df['average'], linestyle='solid')
    ax.fill_between(df['threshold'], df['average'] - df['std'], df['average'] + df['std'], alpha=0.2)

    # Ajustar el rango del eje y entre 0 y 1
    plt.ylim(0, 1)

    # Mostrar el gráfico
    plt.savefig(f'F1-Score_grafico_{name}.pdf')  

elif tipo_grafica == 'overlap':
    # Cargar el archivo CSV
    df = pd.read_csv(f'resultados_overlap_{name}.csv')

    # Obtener las columnas del DataFrame
    columnas = df.columns
    df['threshold'] = [i / 10.0 for i in range(1, 13)]

    fig,ax = plt.subplots(figsize=(8,6))
    plt.rcParams['legend.title_fontsize'] = 18

    ax.set_title('Cantidade de solapamentos umbral euclídea',fontsize=16)
    ax.set_xlabel('Umbral',fontsize=14)
    ax.set_ylabel('Número de solapamentos',fontsize=14)

    plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
    ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    plt.xticks(rotation = 45)
    for label in ax.get_xticklabels():
        label.set_fontsize(9)
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        
    # Calculas la media y la desviacion típica de las columnas
    df['average'] = df[columnas].mean(axis=1)
    df['std'] = df[columnas].std(axis=1)

    # Se representa los valores medios y la desviación del eje X
    ax.plot(df['threshold'], df['average'], linestyle='solid')
    ax.fill_between(df['threshold'], df['average'] - df['std'], df['average'] + df['std'], alpha=0.2)

    # Ajustar el rango del eje y entre 0 y 1
    #plt.ylim(0, 15)

    # Mostrar el gráfico
    plt.savefig(f'Overlap_grafico_{name}.pdf')  

else:
    print("Tipo de gráfica no reconocido")
    sys.exit(1)
