"""
Script con las funciones basicas del Algoritmo Genético.
Funciones incluidas:
- crear_individuo
- seleccion_custom
- crossover
- mutacion
- mut_null
- algoritmo_genetico
- plot_ag_results
"""

import numpy as np
import pandas as pd
import math
import random
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ------------------------------
# Funciones básicas del AG
# ------------------------------

def crear_individuo(n, LEGO):
    """Crea un individuo de tamaño n a partir de LEGO sin reemplazo."""
    if n > len(LEGO):
        raise ValueError("n no puede ser mayor que el tamaño de LEGO")
    return list(np.random.choice(LEGO, size=n, replace=False))

def seleccion(poblacion, fitness_func, modo=1, **fitness_kwargs):
    """Selección por torneo 1v1 usando cualquier función de fitness."""
    ind1, ind2 = random.sample(poblacion, 2)
    fit1 = fitness_func(ind1, **fitness_kwargs)
    fit2 = fitness_func(ind2, **fitness_kwargs)
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
    """Mutación aleatoria de genes en el cromosoma según MUT_RATE."""
    cromosoma = list(cromosoma)
    for i in range(len(cromosoma)):
        if random.random() < MUT_RATE:
            opciones = list(set(LEGO) - set(cromosoma))
            if opciones:
                cromosoma[i] = random.choice(opciones)
    return cromosoma

def mut_null(cromosoma, MUT_NULL_RATE):
    """Reemplaza genes por valor nulo (-1) según MUT_NULL_RATE."""
    cromosoma = list(cromosoma)
    for i in range(len(cromosoma)):
        if random.random() < MUT_NULL_RATE:
            cromosoma[i] = -1
    return cromosoma


# ------------------------------
# Función de fitness
# ------------------------------

def fitness(cromosoma, df_expresion, target, nombres_columnas,
            modelo="rf", metric="accuracy", random_state=42, cv=5):
    """
    Evalúa un cromosoma (lista de índices) seleccionando columnas en df_expresion
    según los nombres dados en nombres_columnas.

    Args:
        cromosoma (list[int]): índices de genes seleccionados.
        df_expresion (pd.DataFrame): matriz de expresión génica (genes en columnas).
        target (pd.DataFrame): dataframe con columnas [ID, clase].
        nombres_columnas (list[str]): lista de nombres de columnas de df_expresion.
        modelo (str): "rf" (Random Forest) o "svm".
        metric (str): métrica de evaluación si se usa SVM.
        random_state (int): semilla para reproducibilidad.
        cv (int): número de folds en validación cruzada (para SVM).
    """

    # Validar índices
    selected_idx = [i for i in cromosoma if isinstance(i, int) and 0 <= i < len(nombres_columnas)]
    if len(selected_idx) == 0:
        return 0.0

    # Obtener los nombres de los genes seleccionados
    genes = [nombres_columnas[i] for i in selected_idx]

    # Validar que existan en df_expresion
    genes_validos = [g for g in genes if g in df_expresion.columns]
    if len(genes_validos) == 0:
        return 0.0

    # --- Alinear target ---
    id_col, class_col = target.columns[0], target.columns[1]
    try:
        if set(df_expresion.index).intersection(set(target[id_col])):
            target_series = target.set_index(id_col)[class_col].reindex(df_expresion.index)
        elif len(target) == len(df_expresion):
            target_series = target.iloc[:, 1].reset_index(drop=True)
            target_series.index = df_expresion.index
        else:
            raise ValueError("No puedo alinear `target` con `df_expresion`.")
    except Exception as e:
        raise ValueError(f"Error alineando target: {e}")

    mask_valid = target_series.notna()
    if mask_valid.sum() == 0:
        return 0.0

    X = df_expresion.loc[mask_valid, genes_validos]
    y = target_series.loc[mask_valid]

    # --- Codificación de etiquetas ---
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    if len(np.unique(y_enc)) != 2:
        return 0.0

    # --- Escalado ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # --- Modelos ---
    if modelo == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            oob_score=True,
            bootstrap=True
        )
        clf.fit(X_scaled, y_enc)
        score = clf.oob_score_

    elif modelo == "svm":
        clf = SVC(kernel="linear", probability=True, random_state=random_state)
        scoring = metric if metric in ["accuracy", "f1", "roc_auc"] else "accuracy"
        scores = cross_val_score(clf, X_scaled, y_enc, cv=cv, scoring=scoring)
        score = np.mean(scores)

    else:
        raise ValueError("Modelo no soportado: 'rf' o 'svm'.")

    return float(score)

# ------------------------------
# Algoritmo Genético
# ------------------------------

def algoritmo_genetico(fitness_func, modo, LEGO, IND_LEN, POP_SIZE, POP_NUM, GENS,
                       MUT_RATE=0.1, MUT_NULL_RATE=0.1, ELITISM=0.1, **fitness_kwargs):
    """
    Ejecuta un algoritmo genético con cualquier función de fitness parametrizable.

    Args:
        fitness_func: función de fitness (primer argumento = cromosoma).
        modo: 1 = maximización, 0 = minimización.
        LEGO: lista de genes posibles.
        IND_LEN: tamaño de cada individuo.
        POP_SIZE: tamaño de cada población.
        POP_NUM: número de poblaciones paralelas.
        GENS: número de generaciones.
        MUT_RATE: tasa de mutación normal.
        MUT_NULL_RATE: tasa de mutación nula (-1).
        ELITISM: fracción de individuos elitistas.
        **fitness_kwargs: parámetros extra para fitness_func.
    
    Returns:
        df_historial: DataFrame con estadísticas por generación.
        mejores_por_poblacion: lista de mejores individuos de cada población final.
        mejor_global: mejor individuo global.
        parametros: diccionario con parámetros usados.
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
            if modo == 1:
                elite = sorted(poblacion, key=lambda ind: fitness_func(ind, **fitness_kwargs), reverse=True)[:n_elite]
            else:
                elite = sorted(poblacion, key=lambda ind: fitness_func(ind, **fitness_kwargs))[:n_elite]
            nueva_poblacion = elite.copy()

            while len(nueva_poblacion) < POP_SIZE:
                padre1 = seleccion(poblacion, fitness_func, modo, **fitness_kwargs)
                padre2 = seleccion(poblacion, fitness_func, modo, **fitness_kwargs)
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

        # Estadísticas por generación
        fitness_vals = [fitness_func(ind, **fitness_kwargs) for poblacion in poblaciones for ind in poblacion]
        historial.append({
            "generacion": gen,
            "fitness_promedio": np.mean(fitness_vals),
            "fitness_mejor": np.max(fitness_vals) if modo == 0 else np.min(fitness_vals),
            "fitness_peor": np.min(fitness_vals) if modo == 1 else np.max(fitness_vals)
        })

    # Mejores de la última generación
    mejores_por_poblacion = [max(poblacion, key=lambda ind: fitness_func(ind, **fitness_kwargs)) for poblacion in poblaciones]
    mejor_global = max(mejores_por_poblacion, key=lambda ind: fitness_func(ind, **fitness_kwargs))
    best_fitness = fitness_func(mejor_global, **fitness_kwargs)


    df_historial = pd.DataFrame(historial)

    parametros = {
        "Tamaño del Individuo": IND_LEN,
        "Tamaño de la Población": POP_SIZE,
        "Número de Poblaciones": POP_NUM,
        "Número de Generaciones": GENS,
        "Tasa de Mutación": MUT_RATE,
        "Tasa de Mutación Nula": MUT_NULL_RATE,
        "Elitismo": ELITISM,
        "# de Genes Disponibles": len(LEGO)
    }


    return df_historial, mejores_por_poblacion, mejor_global, best_fitness, parametros 

# ------------------------------
# Funciones de Visualización
# ------------------------------

def plot_ag_results(df_historial, mejor_global, best_fitness, parametros, nombres_columnas=None):
    """
    Visualiza los resultados de un Algoritmo Genético.

    Args:
        df_historial (DataFrame): historial con columnas ["generacion", "fitness_promedio", "fitness_mejor"].
        mejor_global (array-like): genoma del mejor individuo.
        fitness_valor (float): valor de fitness del mejor individuo.
        parametros (dict): parámetros usados en el AG.
        nombres_columnas (list, opcional): nombres asociados a los índices del genoma.
    """

    # --- 1. Fitness promedio y mejor ---
    plt.figure(figsize=(8, 5))
    plt.plot(df_historial["generacion"], df_historial["fitness_promedio"],
             label='Fitness promedio', color='black')
    plt.plot(df_historial["generacion"], df_historial["fitness_mejor"],
             label='Fitness mejor', color='red', linestyle="--")
    plt.title('Fitness promedio y mejor por generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

    # --- 2. Mejor individuo ---
    plt.figure(figsize=(8, 3))
    plt.axis("off")

    genoma_str = ", ".join(map(str, mejor_global))
    plt.text(0.01, 0.75, f"Mejor individuo (codificado):\n[{genoma_str}]",
             fontsize=10, ha="left")

    if nombres_columnas is not None:
        try:
            columnas_genes = [nombres_columnas[i] for i in mejor_global]
            nombres_genes = ", ".join(columnas_genes)
            plt.text(0.01, 0.5, f"Genes (decodificado):\n[{nombres_genes}]",
                     fontsize=9, ha="left")
        except Exception as e:
            plt.text(0.01, 0.5, f"Error al decodificar nombres de genes: {e}",
                     fontsize=9, ha="left", color='red')

    plt.text(0.01, 0.25, f"Fitness: {best_fitness:.4f}",
             fontsize=12, ha="left", weight="bold")
    plt.tight_layout()
    plt.show()

    # --- 3. Parámetros del AG ---
    plt.figure(figsize=(8, 3))
    plt.axis("off")
    parametros_str = "\n".join([f"{k}: {v}" for k, v in parametros.items()])
    plt.text(0.01, 0.8, "Parámetros del Algoritmo Genético:",
             fontsize=12, ha="left", weight="bold")
    plt.text(0.01, 0.7, parametros_str,
             fontsize=10, ha="left", va="top")
    plt.tight_layout()
    plt.show()

def info_individuo(cromosoma, df):
    """
    Muestra la información de un individuo y estadísticas básicas de las columnas
    indicadas por los índices en el "cromosoma". "df" es el DataFrame de expresión.
    """
    # Normalizar índices válidos
    idxs = [int(i) for i in cromosoma if isinstance(i, (int, np.integer)) and 0 <= int(i) < df.shape[1]]
    if len(idxs) == 0:
        plt.figure(figsize=(6, 2))
        plt.axis("off")
        plt.text(0.01, 0.5, "No hay genes válidos en el cromosoma.", fontsize=10, va="center")
        plt.show()
        return

    # Nombres de columnas correspondientes a los índices
    try:
        nombres = list(df.columns[idxs])
    except Exception:
        nombres = [str(i) for i in idxs]

    nombres_str = ", ".join(map(str, nombres))

    # Subconjunto de df con las columnas seleccionadas
    subset = df.iloc[:, idxs]

    # Estadísticas: no nulos, nulos, media y diferentes de cero (numéricas)
    non_null = subset.notna().sum()
    nulls = subset.shape[0] - non_null
    means = subset.select_dtypes(include=[np.number]).mean()

    non_zero = {}
    for col in subset.columns:
        if np.issubdtype(subset[col].dtype, np.number):
            non_zero[col] = int((subset[col] != 0).sum())
        else:
            non_zero[col] = np.nan

    resumen_lines = []
    for col in subset.columns:
        nn = int(non_null.get(col, 0))
        nz = int(nulls.get(col, 0))
        m = means.get(col, np.nan)
        nzr = non_zero.get(col, np.nan)
        if np.isnan(m):
            resumen_lines.append(f"{col}: no_nulos={nn}, nulos={nz}, distintos_de_cero={nzr}, mean=NA")
        else:
            resumen_lines.append(f"{col}: no_nulos={nn}, nulos={nz}, distintos_de_cero={nzr}, mean={m:.3g}")

    resumen_text = "\n".join(resumen_lines)

    # === Tabla de resumen ===
    fig, ax = plt.subplots(figsize=(10, len(subset.columns) * 0.4 + 2))
    ax.axis("off")

    resumen_df = pd.DataFrame({
        "Gen": subset.columns,
        "No nulos": [int(non_null.get(c, 0)) for c in subset.columns],
        "Nulos": [int(nulls.get(c, 0)) for c in subset.columns],
        "Distintos de cero": [non_zero.get(c, np.nan) for c in subset.columns],
        "Media": [means.get(c, np.nan) for c in subset.columns],
    })

    resumen_df["Media"] = resumen_df["Media"].apply(lambda x: f"{x:.3g}" if pd.notna(x) else "NA")

    tabla = ax.table(cellText=resumen_df.values,
                     colLabels=resumen_df.columns,
                     cellLoc='center',
                     loc='center')

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.2, 1.2)

    # Colores: encabezado con fondo khaki y filas alternadas
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("darkseagreen")
            cell.set_text_props(weight="bold", color="black")
        elif row % 2 == 0:
            cell.set_facecolor("whitesmoke")
        else:
            cell.set_facecolor("lightgrey")

    # Título
    plt.text(0.00, 0.95, "Mejor individuo:", fontsize=11, weight="bold", va="top")
    plt.text(0.00, 0.88, f"[{nombres_str}]", fontsize=9, va="top")

    plt.tight_layout()
    plt.show()

    # === Boxplots ===
    numeric_subset = subset.select_dtypes(include=[np.number])
    if numeric_subset.shape[1] > 0:
        ncols = min(4, numeric_subset.shape[1])  # máximo 4 por fila
        nrows = math.ceil(numeric_subset.shape[1] / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                 figsize=(3 * ncols, 4 * nrows),
                                 sharey=True)

        axes = np.array(axes).reshape(-1)  # Asegurar iterabilidad
        for ax, col in zip(axes, numeric_subset.columns):
            ax.boxplot(numeric_subset[col].dropna(), vert=True)
            ax.set_title(col, fontsize=9)
            ax.tick_params(axis='x', labelrotation=45)

        # Desactivar ejes vacíos
        for ax in axes[len(numeric_subset.columns):]:
            ax.axis("off")

        fig.suptitle("Distribución/Boxplot por Gen", fontsize=12, weight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("No hay columnas numéricas para graficar boxplots.")
