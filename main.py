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
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df2)

        # User inputs for KMedoids model
        n_clusters = st.slider("Banyaknya Cluster", min_value=2, max_value=10, value=3, step=1)
        random_state = st.number_input("Random State", value=0)

        kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state).fit(data)
        ClusLabel = kmedoids.labels_
        nilai_silhouette = silhouette_samples(data, ClusLabel)

        # Visualisasi Silhouette
        Visual = SilhouetteVisualizer(kmedoids, colors='yellowbrick')

     
        Visual.fit(data)

        # Use matplotlib backend to render the plot directly
        st.pyplot(plt.gcf())

        rata_silhouette = silhouette_score(data, ClusLabel)
        st.write(f'Nilai Rata-Rata Silhouette dengan {n_clusters} Cluster : ', rata_silhouette)
        
         # Scatter plot
        st.subheader("Scatter Plot")
        plt.figure(figsize=(8, 6))
        for i in range(n_clusters):
            plt.scatter(data[ClusLabel == i, 0], data[ClusLabel == i, 1], label=f'Cluster {i + 1}')
        plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=100, c='black', marker='x',
                    label='Medoids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        st.pyplot(plt.gcf())
        
       
     
        # Line plot for clusters
        st.subheader("Line Plot of Clusters")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_clusters):
            cluster_indices = np.where(ClusLabel == i)[0]
            cluster_data = df.to_numpy()[cluster_indices]
            ax.plot(cluster_indices, cluster_data[:, 0], label=f'Cluster {i + 1}')
        ax.set_xlabel('Data Points')
        ax.set_ylabel('Feature 1')
        ax.legend()
        st.pyplot(fig)

          # Calculate correlation and display correlation matrix
        st.subheader("Correlation Matrix")
        st.write(df.corr())

        # Display correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
        

elif menu_select == 'Dataset':
    st.title('Dataset')
    df = load_data('Final16.xlsx')
    st.write("Original Dataset:")
    st.write(df)

elif menu_select == 'About':
    st.title('About')
    st.write('This is a Streamlit application for KMedoids clustering.')