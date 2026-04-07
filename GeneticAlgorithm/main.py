"""
Este script contiene el codigo principal para ejecutar un Algoritmo Genético. 

Funcion "fitness" que calcula la aptitutd de un individuo. Recibe al indivudo y entrena un modelo de clasificación con las informacion de l individuo y devuelve como fitness 
alguna metrica de evaluación del modelo. En este caso los datos de expresion y etiquetas se le pasan en archivos csv previamente procesados.
Recibe como parametros:
    cromosoma: individuo (lista de los indices de los genes que conforman al individuo)
    df_expresion: DataFrame con las muestras en filas y los genes en columnas.
    target: DataFrame con dos columnas, donde la primera contiene los identificadores de las muestras y la segunda la clase a la ue pertenece.
    modelo: el modelo de clasificación que se utilizará ("rf" para Random Forest o "svm" para Support Vector Machine).
    metric: la métrica de evaluación que se utilizará para el modelo("accuracy", "f1" o "roc_auc").

Funcion "evaluar_mejor_individuo" que evalua el mejor individuo obtenido del AG con una evaluación más robusta:

Define una lista con un tamaño equivalente a la al numero de genes disponibles para utilisarse.

Llama a la funcion "algoritmo_genetico" del script "algoritmo_genetico.py" para ejecutar el AG, y pasa los parametros necesarios incluyendo la función de fitness definida anteriormente.
Llama a la funcion "plot_ag_results" del script "algoritmo_genetico.py" para visualizar el promedio global y el fitness del mejor individuo en un solo plot, asi como los parametros utilizados en el AG


    
Las funciones adicionales del AG se encuentran en el acripto "algoritmo_genetico.py", el cual incluye:
-Funciones basicas:
    Crear_individuo
    Seleccion
    Crossover
    Mutacion
    Mut_null
-Funcion de fitness
    fitness
- Flujo principal del  AG
    algoritmo_genetico
-Funciones de visualización
    plot_ag_results
"""


import algoritmo_genetico as ag

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ------------------------------
# Definir genes disponibles
# ------------------------------
LEGO = list(range(4950))  # ajusta según tu dataset
# df_expresion = 
# target = 

# ------------------------------
# Ejecutar el Algoritmo Genético
# ------------------------------
df_historial, mejores, mejor_global, parametros = ag.algoritmo_genetico(
    fitness_func=ag.fitness,
    modo=1,
    LEGO=LEGO,
    IND_LEN=20,
    POP_SIZE=20,
    POP_NUM=20,
    GENS=50,
    MUT_RATE=0.1,
    MUT_NULL_RATE=0.1,
    ELITISM=0.1,
    df_expresion="df_expresion.csv",
    target="target.csv",
    modelo="rf",
    metric="accuracy",
    test_size=0.2,
    random_state=42
)

# ------------------------------
# Visualización de resultados
# ------------------------------
ag.plot_ag_results(df_historial, mejor_global, ag.fitness, parametros,
                   df_expresion="df_expresion.csv", target="target.csv", modelo="rf", metric="accuracy")
