import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler  # puede cambiarse por RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


RANDOM_STATE = 42
FEATURE_COLS = ["Evaluacion", "Amabilidad", "Asesorias", "Rapidez", "Emocion BETO"]

PATH_BIB = "biblioteca.csv"
PATH_TEC = "tecnologias.csv"

# 📁 Nueva carpeta de salida
OUT_DIR = Path("salidas_dbscan_pca")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def elegir_mejor_parametros_dbscan(X_scaled, eps_values=None, min_samples_values=None):
    if eps_values is None:
        eps_values = np.linspace(0.1, 0.8, 15)
    if min_samples_values is None:
        min_samples_values = [2, 3, 4, 5]

    mejor_score = -1
    mejor_eps, mejor_min_samples = None, None

    for eps in eps_values:
        for ms in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) <= 1 or len(set(labels)) == len(X_scaled):
                continue
            try:
                score = silhouette_score(X_scaled, labels)
                if score > mejor_score:
                    mejor_score = score
                    mejor_eps, mejor_min_samples = eps, ms
            except Exception:
                continue
    return mejor_eps, mejor_min_samples, mejor_score


def graficar_pca_scatter(X_scaled, labels, titulo, out_path_png):
    if len(np.unique(labels)) < 2 or X_scaled.shape[0] < 2:
        return

    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca_vis.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = "Ruido" if label == -1 else f"Cluster {label}"
        if label == -1:
            color = "lightgray"
        plt.scatter(Z[mask, 0], Z[mask, 1], c=[color], edgecolor="k", s=60, alpha=0.8, label=label_name)

    plt.title(titulo, fontsize=11)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(out_path_png, dpi=150)
    plt.close()

    return pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Cluster": labels})


def clusterizar_subconjunto_dbscan(subset: pd.DataFrame,
                                   instancia: str,
                                   categoria_val,
                                   out_dir: Path):
    X = subset[FEATURE_COLS].to_numpy()

    # Escalado
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # conserva 95% de la varianza
    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xs)

    # Busca mejores parámetros sobre los datos reducidos
    eps_opt, ms_opt, sil = elegir_mejor_parametros_dbscan(X_pca)
    if eps_opt is None:
        eps_opt, ms_opt, sil = 0.8, 3, np.nan

    model = DBSCAN(eps=eps_opt, min_samples=ms_opt)
    labels = model.fit_predict(X_pca)

    subset = subset.copy()
    subset.loc[:, "Cluster"] = labels

    # Resumen (excluye ruido)
    clusters_validos = subset[subset["Cluster"] != -1]
    if len(clusters_validos) > 0:
        resumen = clusters_validos.groupby("Cluster")[FEATURE_COLS].mean().round(3)
        conteos = subset["Cluster"].value_counts().sort_index()
    else:
        resumen = pd.DataFrame()
        conteos = pd.Series(dtype=int)

    base_name = f"{instancia}_cat{categoria_val}"
    subset.to_csv(out_dir / f"{base_name}_etiquetado.csv", index=False, encoding="utf-8")
    resumen.to_csv(out_dir / f"{base_name}_resumen_clusters.csv", encoding="utf-8")
    conteos.to_csv(out_dir / f"{base_name}_conteos_clusters.csv", header=["n"], encoding="utf-8")

    # PCA visual 2D para graficar
    pca_df = graficar_pca_scatter(
        X_scaled=X_pca,
        labels=labels,
        titulo=f"{instancia} - Categoría {categoria_val}",
        out_path_png=out_dir / f"{base_name}_pca2d.png"
    )
    if pca_df is not None:
        pca_df.to_csv(out_dir / f"{base_name}_pca2d.csv", index=False, encoding="utf-8")

    return subset, resumen, conteos, eps_opt, ms_opt, sil, pca.explained_variance_ratio_.sum()


def analizar_instancia_dbscan(df: pd.DataFrame, nombre_instancia: str, out_dir: Path):
    resultados = {}
    categorias = df["Categoria"].dropna().unique()

    for cat in sorted(categorias):
        subset = df.loc[df["Categoria"] == cat].copy()
        faltan = [c for c in FEATURE_COLS if c not in subset.columns]
        if faltan or subset.shape[0] == 0:
            continue

        etiquetado, resumen, conteos, eps, ms, sil, varianza = clusterizar_subconjunto_dbscan(
            subset, nombre_instancia, cat, out_dir
        )

        resultados[cat] = {
            "df_etiquetado": etiquetado,
            "resumen_clusters": resumen,
            "conteos_clusters": conteos,
            "eps": eps,
            "min_samples": ms,
            "silhouette": sil,
            "varianza_explicada": varianza
        }

    return resultados

bib = pd.read_csv(PATH_BIB)
tec = pd.read_csv(PATH_TEC)

res_bib = analizar_instancia_dbscan(bib, "Biblioteca", OUT_DIR)
res_tec = analizar_instancia_dbscan(tec, "Tecnologias", OUT_DIR)

rows = []
for instancia, res in [("Biblioteca", res_bib), ("Tecnologias", res_tec)]:
    for cat, info in res.items():
        rows.append({
            "Instancia": instancia,
            "Categoria": cat,
            "eps": info["eps"],
            "min_samples": info["min_samples"],
            "silhouette": info["silhouette"],
            "VarianzaExplicada_PCA": round(info["varianza_explicada"], 3)
        })
reporte_maestro = pd.DataFrame(rows).sort_values(["Instancia", "Categoria"])
reporte_maestro.to_csv(OUT_DIR / "reporte_maestro_dbscan_pca.csv", index=False, encoding="utf-8")

print("Listo. Ccarpeta:", OUT_DIR.resolve())
