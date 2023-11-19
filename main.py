import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Function to load data
def load_data(file):
    data = pd.read_excel(file, engine='openpyxl')  # Use engine='openpyxl' for reading .xlsx files
    return data

# Sidebar
st.sidebar.title("Menu")
menu_select = st.sidebar.radio("Go to", ('Halaman Utama','Eksperimen', 'Dataset', 'About'))

# Menu Utama
if menu_select == 'Halaman Utama':
    st.title('Halaman Utama')
    # Pendahuluan
    st.subheader("Pendahuluan")
    st.write("Kalimantan, merupakan sebuah pulau yang luas dan beragam di Indonesia, terkenal akan keanekaragaman hayatinya dan ekosistem yang unik. Pulau ini mengalami berbagai fenomena meteorologi yang membentuk iklim dan pola cuaca. Dalam artikel ini, kita akan menjelajahi aspek-aspek meteorologi Kalimantan, mengeksplorasi faktor-faktor seperti curah hujan, variasi suhu, dan pengaruh iklim.")

    st.subheader("Dinamika Suhu")
    st.write("Iklim Khatulistiwa: Proksimitas Kalimantan dengan garis khatulistiwa berkontribusi pada iklim khatulistiwa pulau ini. Akibatnya, suhu cenderung relatif konsisten sepanjang tahun. Suhu siang biasanya berkisar antara 25°C hingga 33°C, memberikan lingkungan yang hangat dan lembab. Malam biasanya lebih sejuk tetapi tetap relatif lembut dibandingkan dengan daerah yang lebih jauh dari garis khatulistiwa.")
    
    st.subheader("Percobaan")
    st.write("Pada penelitian ini dilakukan percobaan untuk mengelompokkan data menjadi clustering agar dapat lebih memahami pola data iklim di Pulau Kalimantan.")
    st.write("Dari 24 Kota yang terdapat stasiun meteorologi di Pulau Kalimantan, hanya 17 kota yang dilakukan percobaan.")
    df1 = load_data('Book1.xlsx')
    st.write("Kota - kota yang digunakan adalah sebagai berikut :")
    st.write(df1)
    
    st.subheader("Hasil")
    st.write("Karena keterbatasan perangkat keras, penelitian hanya bisa menggunakan 6 parameter data saja. Parameter yang digunakan adalah temperatur minimum, maksimum, rata - rata, kelembapan, kecepatan angin maksimum, kecapatan angin rata - rata.")
    st.write("Berikut adalah pola data tahunan dari 6 parameter yang digunakan dalam penelitian :")
    st.image("Tahunan.jpg")
    
    st.write("Pola data setiap bulan yang dihasilkan dari penelitian adalah sebagai berikut : ")
    st.image("Bulanan.jpg")
    
    st.write("Dari hasil penelitian didapatkan perbandingan pola data antar cluster. Perbandingan pola data antar cluster dapat dilihat pada gambar berikut : ")
    st.image("Perbandingan.jpg")
    
    
# Eksperimen
elif menu_select == 'Eksperimen':
    st.title('Eksperimen')
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
    st.write("Dataset yang digunakan sebaiknya merupakan dataset yang telah melalui proses agregasi fitur. Berikut adalah susunan dari dataset yang baik untuk digunakan dalam eksperimen")
    st.write(df)
    
     # Create a buffer to store the Excel file
    excel_buffer = io.BytesIO()

    # Save the DataFrame to the buffer as an Excel file
    df.to_excel(excel_buffer, index=False, engine="openpyxl")

  # Create a download button
    st.download_button(
        label="Unduh Dataset (XLSX)",
        data=excel_buffer,
        file_name="Final16.xlsx",
        key="download_button"
    )
    
elif menu_select == 'About':
    st.title('About')
    st.write('This is a Streamlit application for KMedoids clustering.')
