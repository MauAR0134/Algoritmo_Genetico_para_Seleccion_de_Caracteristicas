import numpy as np
from tdqm import trange

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

def fitness(cromosoma):
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
    fit1 = fitness(ind1)
    fit2 = fitness(ind2)
    if modo == 1:
        return ind1 if fit1 >= fit2 else ind2
    else:
        return ind1 if fit1 <= fit2 else ind2

def crossover(padre1, padre2):
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
    Para cada elemento del cromosoma, con probabilidad MUT_RATE,
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

def algoritmo_genetico(modo):
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
            n_elite = int(ELITISM * POP_SIZE)
            if modo == 1:  # maximización
                elite = sorted(poblacion, key=fitness, reverse=True)[:n_elite]
            else:  # minimización
                elite = sorted(poblacion, key=fitness)[:n_elite]
            nueva_poblacion = elite.copy()

            while len(nueva_poblacion) < POP_SIZE:
                padre1 = seleccion(poblacion, modo)
                padre2 = seleccion(poblacion, modo)
                hijo1, hijo2 = crossover(padre1, padre2)
                hijo1 = mutacion(hijo1, LEGO, MUT_RATE)
                hijo1 = mut_null(hijo1, MUT_NULL_RATE)
                hijo2 = mutacion(hijo2, LEGO, MUT_RATE)
                hijo2 = mut_null(hijo2, MUT_NULL_RATE)
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < POP_SIZE:
                    nueva_poblacion.append(hijo2)
            nuevas_poblaciones.append(nueva_poblacion)
        poblaciones = nuevas_poblaciones

        # Guarda información de cada individuo de la generación
        for i, poblacion in enumerate(poblaciones):
            for ind in poblacion:
                historial.append({
                    "generacion": gen,
                    "poblacion": i,
                    "individuo": ind,
                    "fitness": fitness(ind)
                })

    # Mejores de la última generación
    mejores_por_poblacion = []
    for i, poblacion in enumerate(poblaciones):
        mejor = max(poblacion, key=fitness)
        mejores_por_poblacion.append(mejor)

    mejor_global = max(mejores_por_poblacion, key=fitness)

    # Crea DataFrame con historial
    df_historial = pd.DataFrame(historial)
    # Agrega resumen de parámetros
    parametros = {
        "IND_LEN": IND_LEN,
        "POP_SIZE": POP_SIZE,
        "POP_NUM": POP_NUM,
        "GENS": GENS,
        "MUT_RATE": MUT_RATE,
        "MUT_NULL_RATE": MUT_NULL_RATE,
        "ELITISM": ELITISM,
        "LEGO_LEN": len(LEGO)
    }
    return df_historial, mejores_por_poblacion, mejor_global, parametros

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
    fitness_por_gen = df_historial.groupby("generacion")["fitness"]
    fitness_promedio = fitness_por_gen.mean()
    fitness_maximo = fitness_por_gen.max()

    plt.figure(figsize=(8,5))
    plt.plot(fitness_promedio, label='Fitness promedio', color='black')
    plt.plot(fitness_maximo, label='Fitness máximo', color='red', linestyle="--")
    plt.title('Fitness promedio y máximo por generación')
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
    plt.figure(figsize=(10, 3))
    plt.axis("off")  # quitar ejes

    genoma_str = ", ".join(map(str, mejor_global))
    plt.text(0.01, 0.6, f"Mejor individuo:\n[{genoma_str}]", fontsize=10, ha="left")
    plt.text(0.01, 0.3, f"Fitness: {fitness(mejor_global):.2f}", fontsize=12, ha="left", weight="bold")
    

    plt.show()
    
    # ----------------------------
    # 3. Parámetros del AG
    # ----------------------------
    parametros_md = "\n".join([f"- **{k}**: `{v}`" for k, v in parametros.items()])
    display(Markdown(f"### Parámetros del Algoritmo Genético\n{parametros_md}"))
# ------------------------------
# Configuración de los parámetros del algoritmo
# ------------------------------
POP_SIZE = 20      # Tamaño de la población
POP_NUM = 20       # Número de poblaciones
IND_LEN = 20  # Longitud del individuo
GENS = 200         # Número de generaciones
MUT_RATE = 0.1    # Probabilidad de mutación
MUT_NULL_RATE = 0.1  # Probabilidad de mutación nula
ELITISM = 0.1     # Porcentaje de elitismo
LEGO = list(range(4950))


# Ejecuta el algoritmo y genera las visualizaciones
df_historial, mejores_por_poblacion, mejor_global, parametros = algoritmo_genetico(modo=1)


plot_ag_results(df_historial, mejor_global, fitness, parametros)
