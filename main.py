import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file):
    data = pd.read_excel(file, engine='openpyxl')  # Use engine='openpyxl' for reading .xlsx files
    return data

# Sidebar
st.sidebar.title("Menu")
menu_select = st.sidebar.radio("Go to", ('Halaman Utama', 'Dataset', 'About'))

# Main content
if menu_select == 'Halaman Utama':
    st.title('Halaman Utama')
    uploaded_file = st.file_uploader("Upload Excel", type=["xls", "xlsx"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df2 = df.transpose()

        # Scaling method option
        scaling_method = st.checkbox("Gunakan Metode Skala", value=True)
        if scaling_method:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df2)
        else:
            data = df2.to_numpy()

        # User inputs for KMedoids model
        n_clusters = st.slider("Banyaknya Cluster", min_value=2, max_value=10, value=3, step=1)
        random_state = st.number_input("Random State", value=0)
        max_iter = st.number_input("Max Iterations", value=300, step=10)  # Tambahkan opsi untuk max_iter

        # KMedoids method option
        method_options = st.selectbox("Metode KMedoids", ["pam", "alternate"])
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, method=method_options).fit(data)
        ClusLabel = kmedoids.labels_
        nilai_silhouette = silhouette_samples(data, ClusLabel)

        # Visualisasi Silhouette
        st.subheader("Visualisasi Silhouette")
        Visual = SilhouetteVisualizer(kmedoids, colors='yellowbrick')
        Visual.fit(data)
        st.pyplot(plt.gcf())

        # Rata-rata Silhouette Score
        rata_silhouette = silhouette_score(data, ClusLabel)
        st.write(f'Nilai Rata-Rata Silhouette dengan {n_clusters} Cluster : ', rata_silhouette)

        # DataFrame dengan Cluster Labels
        df2_labeled = df2.copy()
        df2_labeled['Cluster'] = ClusLabel

        # Tampilkan hanya kolom 'Cluster' pada dataset
        st.subheader("Kota dan Cluster")
        cluster_column = df2_labeled['Cluster']
        st.write(cluster_column)

        # Scatter plot
        st.subheader("Scatter Plot")
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(data[ClusLabel == i, 0], data[ClusLabel == i, 1], label=f'Cluster {i + 1}')
        plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=100, c='black', marker='x',
                    label='Medoids')
        plt.title('Scatter Plot of Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        st.pyplot(plt.gcf())

        # Line plot for clusters
        st.subheader("Line Plot of Clusters")
        for i in range(n_clusters):
            cluster_indices = np.where(ClusLabel == i)[0]
            cluster_data = df.to_numpy()[cluster_indices]

            # Sorting the data for plotting
            sorted_indices = np.argsort(cluster_data[:, 0])
            sorted_cluster_data = cluster_data[sorted_indices]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sorted_cluster_data[:, 0], marker='o', linestyle='-', label=f'Cluster {i + 1}', linewidth=2, markersize=8)
            ax.set_xlabel('Sorted Data Points', fontsize=12)
            ax.set_ylabel('Feature 1', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.subplots_adjust(wspace=0.3, hspace=0.5)
            st.pyplot(fig)

        # Display correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

       # # Interaktif dengan Pemilihan Fitur
        # st.subheader("Analisis Berdasarkan Fitur")
        # feature_selection = st.multiselect("Pilih Fitur untuk Analisis", df.columns)
        # for feature in feature_selection:
        #     if df[feature].dtype != 'O':  # Skip non-numeric columns
        #         plt.figure(figsize=(8, 6))
        #         sns.boxplot(x='Cluster', y=feature, data=df2_labeled)
        #         plt.title(f"Boxplot {feature} Berdasarkan Cluster")
        #         st.pyplot(plt.gcf())
        #     else:
        #         st.warning(f"Fitur '{feature}' tidak dapat diplot karena bukan tipe data numerik.")

        # # Eksplorasi Lebih Lanjut pada Setiap Cluster
        # st.subheader("Eksplorasi Lebih Lanjut pada Setiap Cluster")
        # for i in range(n_clusters):
        #     st.subheader(f"Analisis Cluster {i + 1}")
        #     cluster_indices = np.where(ClusLabel == i)[0]
        #     cluster_data = df.iloc[cluster_indices].drop('Cluster', axis=1, errors='ignore')
        #     st.write(cluster_data.describe())

        # # Visualisasi Distribusi Fitur
        # st.subheader("Distribusi Fitur")
        # for feature in df.columns:
        #     plt.figure(figsize=(8, 6))
        #     sns.histplot(df[feature], kde=True)
        #     plt.title(f"Distribusi {feature}")
        #     st.pyplot(plt.gcf())

        # # Filter Berdasarkan Range Nilai
        # st.subheader("Filter Berdasarkan Range Nilai")
        # selected_feature = st.selectbox("Pilih Fitur untuk Filter", df.columns)
        # min_value = st.number_input(f"Min {selected_feature}", value=df[selected_feature].min())
        # max_value = st.number_input(f"Max {selected_feature}", value=df[selected_feature].max())
        # filtered_data = df[(df[selected_feature] >= min_value) & (df[selected_feature] <= max_value)]
        # st.write(filtered_data)


elif menu_select == 'Dataset':
    st.title('Dataset')
    df = load_data('Final16.xlsx')
    st.write("Original Dataset:")
    st.write(df)

elif menu_select == 'About':
    st.title('About')
    st.write('This is a Streamlit application for KMedoids clustering.')
