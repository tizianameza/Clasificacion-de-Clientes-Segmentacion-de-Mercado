# Autor: Tiziana Meza
# Fecha: Feb-2024
# Descripción:Este script es un ejemplo de segmentación de mercado de clientes utilizando el algoritmo K-Means en Python.
# Versión de Python: 3.6
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar datos de transacciones históricas de clientes
data = pd.read_csv("datos_transacciones_clientes.csv")

# Preprocesamiento de datos
X = data.drop(columns=['ClienteID'])  # Eliminar la columna de identificación del cliente
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamiento del modelo de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Asignar etiquetas de clusters a los datos
data['Cluster'] = kmeans.labels_

# Análisis de los clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_center_df = pd.DataFrame(cluster_centers, columns=X.columns)
print("Centros de los clusters:")
print(cluster_center_df)

# Visualización de los clusters en un gráfico de dispersión (2D)
plt.scatter(data['Cantidad_Compras'], data['Monto_Total'], c=data['Cluster'], cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, c='red', label='Centros de los clusters')
plt.xlabel('Cantidad de Compras')
plt.ylabel('Monto Total')
plt.title('Segmentación de Mercado de Clientes')
plt.legend()
plt.show()
