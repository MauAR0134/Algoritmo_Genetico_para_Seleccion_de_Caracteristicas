import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import random
from tqdm import trange

# ------------------------------
# Funciones básicas del AG
# ------------------------------

def crear_individuo(n, LEGO):
    """
    Devuelve una lista de n elementos tomados aleatoriamente de LEGO sin reemplazo.
    Si n > len(LEGO), lanza un ValueError.
    """
    if n > len(LEGO):
        raise ValueError("n no puede ser mayor que el tamaño de LEGO")
    return list(np.random.choice(LEGO, size=n, replace=False))

def funcion_objetivo(cromosoma):
    """
    Calcula la media de todos los valores del individuo (cromosoma).
    """
    return np.mean(cromosoma)

def seleccion(poblacion, modo=1):
    """
    Selección por torneo 1v1.
    Si modo=1, retorna el de mayor fitness (maximización).
    Si modo=0, retorna el de menor fitness (minimización).
    """
    ind1, ind2 = random.sample(poblacion, 2)
    fit1 = funcion_objetivo(ind1)
    fit2 = funcion_objetivo(ind2)
    if modo == 1:
        return ind1 if fit1 >= fit2 else ind2
    else:
        return ind1 if fit1 <= fit2 else ind2

def crossover(padre1, padre2, IND_LEN):
    """Cruce de un punto."""
    punto = random.randint(1, IND_LEN - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

def mutacion(cromosoma, LEGO, MUT_RATE):
    """
    Para cada elemento del cromosoma, con probabilidad MUT_RATE,
    lo reemplaza por un valor aleatorio de LEGO que no esté ya en el cromosoma.
    """
    cromosoma = list(cromosoma)  # Asegura que es mutable
    for i in range(len(cromosoma)):
        if random.random() < MUT_RATE:
            opciones = list(set(LEGO) - set(cromosoma))
            if opciones:
                cromosoma[i] = random.choice(opciones)
    return cromosoma

def mut_null(cromosoma, MUT_NULL_RATE):
    """
    Para cada elemento del cromosoma, con probabilidad MUT_NULL_RATE,
    lo reemplaza por el valor nulo (-1) en vez de otro valor de LEGO.
    """
    cromosoma = list(cromosoma)  # Asegura que es mutable
    for i in range(len(cromosoma)):
        if random.random() < MUT_NULL_RATE:
            cromosoma[i] = -1
    return cromosoma

# ------------------------------
# Algoritmo Genético
# ------------------------------

def algoritmo_genetico(modo,LEGO, IND_LEN, POP_SIZE, POP_NUM, GENS, MUT_RATE=0.1, MUT_NULL_RATE=0.1, ELITISM=0.1):
    """
    Ejecuta el algoritmo genético y retorna el historial, mejores individuos,
    mejor individuo global y parámetros utilizados.
    """
    poblaciones = [
        [crear_individuo(IND_LEN, LEGO) for _ in range(POP_SIZE)]
        for _ in range(POP_NUM)
    ]

    historial = []

    for gen in trange(GENS, desc="Evolución AG"):
        nuevas_poblaciones = []
        for poblacion in poblaciones:
            n_elite = max(1, int(ELITISM * POP_SIZE))
            if modo == 1:  # maximización
                elite = sorted(poblacion, key=funcion_objetivo, reverse=True)[:n_elite]
            else:  # minimización
                elite = sorted(poblacion, key=funcion_objetivo)[:n_elite]
            nueva_poblacion = elite.copy()

            while len(nueva_poblacion) < POP_SIZE:
                padre1 = seleccion(poblacion, modo)
                padre2 = seleccion(poblacion, modo)
                hijo1, hijo2 = crossover(padre1, padre2, IND_LEN)
                hijo1 = mutacion(hijo1, LEGO, MUT_RATE)
                hijo1 = mut_null(hijo1, MUT_NULL_RATE)
                hijo2 = mutacion(hijo2, LEGO, MUT_RATE)
                hijo2 = mut_null(hijo2, MUT_NULL_RATE)
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < POP_SIZE:
                    nueva_poblacion.append(hijo2)
            nuevas_poblaciones.append(nueva_poblacion)
        poblaciones = nuevas_poblaciones

            # ---- solo estadísticas por generación ----
        fitness_vals = [funcion_objetivo(ind) for poblacion in poblaciones for ind in poblacion]
        historial.append({
        "generacion": gen,
        "fitness_promedio": np.mean(fitness_vals),
        "fitness_mejor": np.max(fitness_vals) if modo == 1 else np.min(fitness_vals),
        "fitness_peor": np.min(fitness_vals) if modo == 1 else np.max(fitness_vals)
        })

    # Mejores de la última generación
    mejores_por_poblacion = []
    for i, poblacion in enumerate(poblaciones):
        mejor = max(poblacion, key=funcion_objetivo)
        mejores_por_poblacion.append(mejor)

    mejor_global = max(mejores_por_poblacion, key=funcion_objetivo)

    # Crea DataFrame con historial
    df_historial = pd.DataFrame(historial)
    # Agrega resumen de parámetros
    parametros = {
        "Tamaño del Individuo": IND_LEN,
        "Tamaño de la Población": POP_SIZE,
        "Número de Poblaciones": POP_NUM,
        "Número de Generaciones": GENS,
        "Tasa de Mutación": MUT_RATE,
        "Tasa de Mutación Nula": MUT_NULL_RATE,
        "Elitismo": ELITISM,
        "LEGO_LEN": len(LEGO)
    }
    return df_historial, mejores_por_poblacion, mejor_global, parametros

# ------------------------------
# Funciones de Visualización
# ------------------------------


## Funcion para graficar resultados

def plot_ag_results(df_historial, mejor_global, fitness, parametros):
    """
    Función para graficar resultados de un algoritmo genético:
    1. Fitness promedio y máximo por generación.
    2. Mejor individuo y su fitness.
    3. Parámetros del algoritmo.
    
    Args:
        df_historial: DataFrame con historial del AG (de algoritmo_genetico).
        mejor_global: Mejor individuo global.
        fitness: Función de fitness usada.
        parametros: Diccionario con los parámetros del AG.
    """
      # ----------------------------
    # 1. Fitness promedio y máximo
    # ----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(df_historial["generacion"], df_historial["fitness_promedio"], 
             label='Fitness promedio', color='black')
    plt.plot(df_historial["generacion"], df_historial["fitness_mejor"], 
             label='Fitness mejor', color='red', linestyle="--")
    plt.title('Fitness promedio y mejor por generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid()
    
    # Ajuste automático de ticks para eje x
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    
    plt.show()

    
    # ----------------------------
    # 2. Mejor individuo
    # ----------------------------
    plt.figure(figsize=(8, 3))
    plt.axis("off")  # quitar ejes

    genoma_str = ", ".join(map(str, mejor_global))
    plt.text(0.01, 0.6, f"Mejor individuo:\n[{genoma_str}]", fontsize=10, ha="left")
    plt.text(0.01, 0.3, f"Fitness: {fitness(mejor_global):.2f}", fontsize=12, ha="left", weight="bold")
    

    plt.show()
   
    # ----------------------------
    # 3. Parámetros del AG
    # ----------------------------
    plt.figure(figsize=(8, 3))
    plt.axis("off")  # quitar ejes

    parametros_str = "\n".join([f"{k}: {v}" for k, v in parametros.items()])
    plt.text(0.1, 0.8, "Parámetros del Algoritmo Genético:", fontsize=12, ha="left", weight="bold")
    plt.text(0.1, 0.7, parametros_str, fontsize=10, ha="left", va="top")

    plt.show()
