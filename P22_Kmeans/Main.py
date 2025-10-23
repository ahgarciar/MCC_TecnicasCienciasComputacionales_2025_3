import matplotlib.pyplot as plt
from datasets import load_dataset
from fontTools.misc.cython import returns
from scipy.cluster.vq import kmeans
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def loadData():
    iris = load_iris()
    X = iris.data[:, :4]
    return X

def Import_CSV(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    X = df.values
    return X

def ElbowMethod(X):
    # probamos diferentes valores de k para visualizar el metodo del codo
    wcss = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        #print(kmeans.labels_)
        wcss.append(kmeans.inertia_)  # WCSS - valor dispersion
        print("K= ", k)
        print("WCSS:", kmeans.inertia_)

    # Graficar el metodo del codo (Elbow Method)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('WCSS')
    plt.title('Método Elbow para encontrar k óptimo')
    plt.show()

def Metrics(X, clusters):
    # WCSS - Inercia
    # Valor bajo - Mejor agrupacion
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    print("WCSS:", kmeans.inertia_)

    # Silhouette
    # Cercano a uno buena separacion
    # Cercano a cero mala separacion
    # Negativo MAL
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("Coeficiente de Silhouette:", silhouette_avg)

    # Davies-Bouldin
    # Bajo - Mejor separacion
    #Alto - Mala separacion
    from sklearn.metrics import davies_bouldin_score
    db_index = davies_bouldin_score(X, kmeans.labels_)
    print("Índice de Davies-Bouldin:", db_index)

def Kmeans_method(X, clusters):
    # Aplicar K-Means con 3 clusters
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_  # Etiquetas de los clusters asignados
    centroids = kmeans.cluster_centers_  # Centros de los clusters

    print("Centroides finales:\n", centroids)
    print("\nAsignacion:\n", labels)

    # Graficar los puntos de datos con colores según sus clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)

    # Graficar los centroides
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')

    plt.title("Clustering con K-Means")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend()
    plt.show()

def Kmeans_method_3D(X, clusters):
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Gráfico 3D para visualizar los clusters
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o', alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroides')

    ax.set_xlabel("Atributo 1")
    ax.set_ylabel("Atributo 2")
    ax.set_zlabel("Atributo 3")
    ax.set_title("Clustering con K-Means (3 Atributos)")
    plt.legend()
    plt.show()

def Kmeans_PCA(X, clusters):
    from sklearn.decomposition import PCA
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Opcional: Reducir a 2D con PCA si hay más de 3 atributos
    if X.shape[1] > 3:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Mostrar los componentes principales
        print("Componentes principales:\n", pca.components_)
        #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
        #print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

        centroids_pca = pca.transform(centroids)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.title("Clusters en 2D usando PCA")
        plt.legend()
        plt.show()

def Kmeans_PCA3D(X, clusters):
    from sklearn.decomposition import PCA
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Opcional: Reducir a 2D con PCA si hay más de 3 atributos
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        # Mostrar los componentes principales
        print("Componentes principales:\n", pca.components_)
        #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
        #print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

        centroids_pca = pca.transform(centroids)

        # Gráfico 3D para visualizar los componentes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis',marker='o', alpha=0.7)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], c='red', marker='X', s=200, label='Centroides')
        ax.set_xlabel("CP 1")
        ax.set_ylabel("CP 2")
        ax.set_zlabel("CP 3")
        ax.set_title("Clustering con K-Means PCA 3D")
        plt.legend()
        plt.show()

def pca():
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data[:, :4]

    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Ver los nuevos datos transformados
    print(X_pca[:5])  # Primeras 5 filas después del PCA


if __name__ == '__main__':
    X=loadData()
    #ElbowMethod(X)
    #kmeans = Kmeans_method(X,3)
    #kmeans = Kmeans_method_3D(X,clusters=3)
    #kmeans = Kmeans_PCA(X, 3)
    #kmeans = Kmeans_PCA3D(X, 3)
    Metrics(X,3)


##indice de davies --- mas bajo es mejor !!
