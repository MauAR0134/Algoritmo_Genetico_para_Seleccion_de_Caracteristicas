import algoritmo_genetico as ag

# Definir la cantidad de genes disponibles
LEGO = list(range(4950))

# Definir la Función Objetivo
def funcion_objetivo(individuo):
    return sum(individuo)

# Ejecutar el Algoritmo Genético
df_historial, mejores_por_poblacion, mejor_global, parametros = ag.algoritmo_genetico(modo=1, IND_LEN=10, POP_SIZE=10, POP_NUM=10, GENS=200, MUT_RATE=0.1, MUT_NULL_RATE=0.1, ELITISM=0.1, LEGO=LEGO)

# Generar las visualizaciones
ag.plot_ag_results(df_historial, mejor_global, funcion_objetivo, parametros)
