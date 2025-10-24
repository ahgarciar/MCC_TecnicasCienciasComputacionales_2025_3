import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


RANDOM_STATE = 42
K_MIN, K_MAX = 2, 6
FEATURE_COLS = ["Evaluacion", "Amabilidad", "Asesorias", "Rapidez", "Emocion BETO"]

PATH_BIB = "biblioteca.csv"
PATH_TEC = "tecnologias.csv"

OUT_DIR = Path("salidas_clusters")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def elegir_mejor_k(X_scaled, k_min=K_MIN, k_max=K_MAX):
    n = X_scaled.shape[0]
    k_max_real = max(k_min, min(k_max, n - 1))
    if n < 3 or k_max_real < 2:
        return (1 if n == 1 else 2, np.nan)

    mejor_k, mejor_score = None, -np.inf
    for k in range(k_min, k_max_real + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(X_scaled)
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = -np.inf
        if score > mejor_score:
            mejor_k, mejor_score = k, score

    if mejor_k is None:
        mejor_k = min(2, max(1, n))
        mejor_score = np.nan
    return mejor_k, mejor_score


def graficar_pca_scatter(X_scaled, labels, titulo, out_path_png):
    if len(np.unique(labels)) < 2 or X_scaled.shape[0] < 2:
        return  # sin suficiente variación para visualizar

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='viridis', alpha=0.85, edgecolor='k', s=60)
    plt.title(titulo, fontsize=11)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")

    # Leyenda bien posicionada fuera del gráfico
    handles, _ = scatter.legend_elements()
    plt.legend(
        handles,
        [f"Cluster {c}" for c in sorted(np.unique(labels))],
        title="Clusters",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # deja espacio a la derecha para la leyenda
    plt.savefig(out_path_png, dpi=150)
    plt.close()

    # También regresa el dataframe PCA por si se quiere guardar
    return pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Cluster": labels})


def clusterizar_subconjunto(subset: pd.DataFrame,
                            instancia: str,
                            categoria_val,
                            out_dir: Path):
    # Selección de variables
    X = subset[FEATURE_COLS].to_numpy()

    # Estandarización
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Elegir k y entrenar
    k_opt, sil = elegir_mejor_k(Xs)
    km = KMeans(n_clusters=k_opt, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(Xs)

    # Asignación segura
    subset = subset.copy()
    subset.loc[:, "Cluster"] = labels

    # Resúmenes
    resumen = subset.groupby("Cluster")[FEATURE_COLS].mean().round(3)
    conteos = subset["Cluster"].value_counts().sort_index()

    # Identificadores de archivo
    base_name = f"{instancia}_cat{categoria_val}"
    # Guardar etiquetado y resúmenes
    subset.to_csv(out_dir / f"{base_name}_etiquetado.csv", index=False, encoding="utf-8")
    resumen.to_csv(out_dir / f"{base_name}_resumen_clusters.csv", encoding="utf-8")
    conteos.to_csv(out_dir / f"{base_name}_conteos_clusters.csv", header=["n"], encoding="utf-8")

    # PCA: CSV + PNG
    pca_df = None
    try:
        pca_df = graficar_pca_scatter(
            X_scaled=Xs,
            labels=labels,
            titulo=f"{instancia} - Categoría {categoria_val}",
            out_path_png=out_dir / f"{base_name}_pca2d.png"
        )
        if pca_df is not None:
            pca_df.to_csv(out_dir / f"{base_name}_pca2d.csv", index=False, encoding="utf-8")
    except Exception:
        pass

    return subset, resumen, conteos, k_opt, sil


def analizar_instancia(df: pd.DataFrame, nombre_instancia: str, out_dir: Path):
    resultados = {}
    categorias = df["Categoria"].dropna().unique()

    for cat in sorted(categorias):
        subset = df.loc[df["Categoria"] == cat].copy()

        faltan = [c for c in FEATURE_COLS if c not in subset.columns]
        if faltan or subset.shape[0] == 0:
            continue

        etiquetado, resumen, conteos, k_opt, sil = clusterizar_subconjunto(
            subset, nombre_instancia, cat, out_dir
        )

        resultados[cat] = {
            "df_etiquetado": etiquetado,
            "resumen_clusters": resumen,
            "conteos_clusters": conteos,
            "k_opt": k_opt,
            "silhouette": sil,
        }

    return resultados

bib = pd.read_csv(PATH_BIB)
tec = pd.read_csv(PATH_TEC)

res_bib = analizar_instancia(bib, "Biblioteca", OUT_DIR)
res_tec = analizar_instancia(tec, "Tecnologias", OUT_DIR)

rows = []
for instancia, res in [("Biblioteca", res_bib), ("Tecnologias", res_tec)]:
    for cat, info in res.items():
        rows.append({
            "Instancia": instancia,
            "Categoria": cat,
            "k_opt": info["k_opt"],
            "silhouette": info["silhouette"]
        })
reporte_maestro = pd.DataFrame(rows).sort_values(["Instancia", "Categoria"])
reporte_maestro.to_csv(OUT_DIR / "reporte_maestro_clusters.csv", index=False, encoding="utf-8")

print("Listo. Carpeta:", OUT_DIR.resolve())
