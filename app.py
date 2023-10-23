import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import io

st.title("Analizador de datos")

st.header("Carga tu archivo CSV:")
data_file = st.file_uploader("CSV", type=["csv"])


if data_file is not None:
    data = pd.read_csv(data_file)


    cln = st.sidebar.checkbox("Limpiar datos")
    plot = st.sidebar.checkbox("Hacer gráficas")
    Pred = st.sidebar.checkbox("Hacer predicciones")

    if cln:
        st.sidebar.header("Limpieza de datos")
        cln_options = ["Valores nulos", "Duplicados"]
        selected_plot = st.sidebar.multiselect("Tratar los datos", cln_options)
        if "Valores nulos" in selected_plot:
            st.write("")
            radio = st.sidebar.radio("Para los valores nulos", ("Eliminar valores nulos", "Reemplazar por la media"))
            if radio == "Eliminar valores nulos":
                data = data.dropna()
            elif radio == "Reemplazar por la media":
                valores_nulos = data.isnull().sum()
                valores_nulos = valores_nulos[valores_nulos > 0].index.tolist()
                columnas_str = data.select_dtypes(include=['object']).columns.tolist()
                modas = data[columnas_str].mode().iloc[0]

                for col in valores_nulos:
                    if col in columnas_str:
                        data[col].fillna(modas[col], inplace=True)
                    else:
                        medias = data[col].mean()
                        data[col].fillna(medias, inplace=True)


        if "Duplicados" in selected_plot:
            data = data.drop_duplicates()



    st.write("Vista general:")
    st.dataframe(data)
    # filas y columnas
    filas, columnas = data.shape
    st.write(f"**Número de filas:** {filas}.<br>**Número de columnas:** {columnas}", unsafe_allow_html=True)
    # valores nulos
    null = data.isnull().sum()
    null = pd.DataFrame(null, columns=["TotalNulos"])
    null = null.transpose()
    st.write("Valores nulos:")
    st.dataframe(null)
    # duplicados
    dup = data.columns
    duplicados = data[data.duplicated()]
    st.write(f"**Total de duplicados**: {len(duplicados)}")

    if plot:
        st.sidebar.header("Graficar datos")
        plot_options = ["Gráfico de barras", "Gráfico de dispersión", "Histograma", "Box plot", "Mapa de calor"]
        selected_plot = st.sidebar.selectbox("Escoja el tipo de gráfico", plot_options)

        if selected_plot == "Gráfico de barras":
            x_axis = st.sidebar.selectbox("Seleccione el eje X", data.columns)
            y_axis = st.sidebar.selectbox("Seleccione el eje Y", data.columns)
            st.write("Gráfico de barras:")
            fig, ax = plt.subplots()

            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

            st.pyplot(fig)

        elif selected_plot == "Gráfico de dispersión":
            x_axis = st.sidebar.selectbox("Seleccione el eje X", data.columns)
            y_axis = st.sidebar.selectbox("Seleccione el eje Y", data.columns)
            st.write("Gráfico de dispersión:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)

            st.pyplot(fig)

        elif selected_plot == "Histograma":
            column = st.sidebar.selectbox("Seleccione la columna", data.columns)
            bins = st.sidebar.slider("Número de intervalos ", 5, 100, 20)
            st.write("Histograma:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Seleccione la columna", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Mapa de calor":
            st.write("Mapa de calor:")
            data1 = pd.get_dummies(data)
            correlation = data1.corr()

            # hacer el top por si hay muchas variables
            if len(data1.columns) > 10:
                n_top = 10
            else:
                n_top = len(data1.columns)
            top_corr = correlation.unstack().sort_values(ascending=False).drop_duplicates()
            top_corr = top_corr.head(n_top)
            # filtrar
            selected_cols = set(col[1] for col in top_corr.index)
            data_corr = data1[list(selected_cols)]

            fig, ax = plt.subplots()
            sns.heatmap(correlation, annot=True, ax=ax,  fmt='.2f', cmap="YlGnBu")
            st.pyplot(fig)

    if Pred:
        st.write("Predicción")
        models = st.sidebar.selectbox("Seleccione el tipo", ("Regresión", "Clasificación"))
        if models == "Regresión":
            from pycaret.regression import *
        elif models == "Clasificación":
            from pycaret.classification import *

        st.write("")
        target = st.sidebar.selectbox("Seleccione columna a predecir", (data.columns))
        exp = setup(data = data, target= target, index=False)
        summary_tab = pull()
        st.write("Tabla resumen:")
        st.write(summary_tab)
        mensaje = st.empty()
        mensaje.text('Generando...')

        best_model = compare_models()
        mensaje.text('Listo')
        summary_model = pull()
        st.write("Resumen de modelos:")
        st.write(summary_model.head())

        plot_model(best_model, verbose=False, save=True)
        fig, ax = plt.subplots()
        img = plt.imread('Residuals.png')
        ax.imshow(img)
        ax.axis('off')
        st.write("Gráfico residuales:")
        st.pyplot(fig)
        plot_model(best_model, plot='feature', verbose=False, save=True)
        fig, ax = plt.subplots()
        img = plt.imread('Feature Importance.png')
        ax.imshow(img)
        ax.axis('off')
        st.write("Gráfico importancia de variables:")
        st.pyplot(fig)

        final_model = finalize_model(best_model)



else:
    st.subheader("No se ha cargado ningún archivo. Mostrando ejemplo aleatorio:")
    # Aquí carga tu ejemplo predeterminado como un DataFrame de pandas
    # Por ejemplo:
    res_columnas = ['ColumnaA', 'ColumnaB', 'ColumnaC']
    data = pd.DataFrame(np.random.randint(0,100,size=(100, 3)), columns=res_columnas)
    data['Target'] = data['ColumnaA'] + 2*data['ColumnaB'] - data['ColumnaC']
    data.loc[1, 'ColumnaB'] = np.nan
    data.loc[5, 'ColumnaC'] = np.nan
    data.loc[7, 'ColumnaA'] = np.nan
    data.iloc[9] = data.iloc[2].values
    data.iloc[8] = data.iloc[3].values


    cln = st.sidebar.checkbox("Limpiar datos")
    plot = st.sidebar.checkbox("Hacer gráficas")
    Pred = st.sidebar.checkbox("Hacer predicciones")

    if cln:
        st.sidebar.header("Limpieza de datos")
        cln_options = ["Valores nulos", "Duplicados"]
        selected_plot = st.sidebar.multiselect("Tratar los datos", cln_options)
        if "Valores nulos" in selected_plot:
            st.write("")
            radio = st.sidebar.radio("Para los valores nulos", ("Eliminar valores nulos", "Reemplazar por la media"))
            if radio == "Eliminar valores nulos":
                data = data.dropna()
            elif radio == "Reemplazar por la media":
                valores_nulos = data.isnull().sum()
                valores_nulos = valores_nulos[valores_nulos > 0].index.tolist()
                columnas_str = data.select_dtypes(include=['object']).columns.tolist()


                for col in valores_nulos:
                    if col in columnas_str:
                        modas = data[columnas_str].mode().iloc[0]
                        data[col].fillna(modas[col], inplace=True)
                    else:
                        medias = data[col].mean()
                        data[col].fillna(medias, inplace=True)

        if "Duplicados" in selected_plot:
            data = data.drop_duplicates()

    st.write("Vista general:")
    st.dataframe(data)
    # filas y columnas
    filas, columnas = data.shape
    st.write(f"**Número de filas:** {filas}.<br>**Número de columnas:** {columnas}", unsafe_allow_html=True)
    # valores nulos
    null = data.isnull().sum()
    null = pd.DataFrame(null, columns=["TotalNulos"])
    null = null.transpose()
    st.write("Valores nulos:")
    st.dataframe(null)
    # duplicados
    dup = data.columns
    duplicados = data[data.duplicated()]
    st.write(f"Total de duplicados: {len(duplicados)}")

    if plot:
        st.sidebar.header("Graficar datos")
        plot_options = ["Gráfico de barras", "Gráfico de dispersión", "Histograma", "Box plot", "Mapa de calor"]
        selected_plot = st.sidebar.selectbox("Escoja el tipo de gráfico", plot_options)

        if selected_plot == "Gráfico de barras":
            x_axis = st.sidebar.selectbox("Seleccione el eje X", data.columns)
            y_axis = st.sidebar.selectbox("Seleccione el eje Y", data.columns)
            st.write("Gráfico de barras:")
            fig, ax = plt.subplots()

            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)

            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

            st.pyplot(fig)

        elif selected_plot == "Gráfico de dispersión":
            x_axis = st.sidebar.selectbox("Seleccione el eje X", data.columns)
            y_axis = st.sidebar.selectbox("Seleccione el eje Y", data.columns)
            st.write("Gráfico de dispersión:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)

            st.pyplot(fig)

        elif selected_plot == "Histograma":
            column = st.sidebar.selectbox("Seleccione la columna", data.columns)
            bins = st.sidebar.slider("Número de intervalos ", 5, 100, 20)
            st.write("Histograma:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=15))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Seleccione la columna", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Mapa de calor":
            st.write("Mapa de calor:")
            data1 = pd.get_dummies(data)
            correlation = data1.corr()

            # hacer el top por si hay muchas variables
            if len(data1.columns) > 10:
                n_top = 10
            else:
                n_top = len(data1.columns)
            top_corr = correlation.unstack().sort_values(ascending=False).drop_duplicates()
            top_corr = top_corr.head(n_top)
            # filtrar
            selected_cols = set(col[1] for col in top_corr.index)
            data_corr = data1[list(selected_cols)]

            fig, ax = plt.subplots()
            sns.heatmap(correlation, annot=True, ax=ax, fmt='.2f', cmap="YlGnBu")
            st.pyplot(fig)


    if Pred:
        st.write("Predicción")

        models = st.sidebar.selectbox("Seleccione el tipo", ("Regresión", "Clasificación"))
        if models == "Regresión":
            from pycaret.regression import *
        elif models == "Clasificación":
            from pycaret.classification import *

        st.write("")
        target = st.sidebar.selectbox("Seleccione columna a predecir", (data.columns))
        exp = setup(data=data, target=target, index=False)
        summary_tab = pull()
        st.write("Tabla resumen:")
        st.write(summary_tab)
        mensaje = st.empty()
        mensaje.text('Generando...')

        best_model = compare_models()
        mensaje.text('Listo')
        summary_model = pull()
        st.write("Resumen de modelos:")
        st.write(summary_model.head())
        
        plot_model(best_model, verbose=False, save=True)
        fig, ax = plt.subplots()
        img = plt.imread('Residuals.png')
        ax.imshow(img)
        ax.axis('off')
        st.write("Gráfico residuales:")
        st.pyplot(fig)
        plot_model(best_model, plot='feature', verbose=False, save=True)
        fig, ax = plt.subplots()
        img = plt.imread('Feature Importance.png')
        ax.imshow(img)
        ax.axis('off')
        st.write("Gráfico importancia de variables:")
        st.pyplot(fig)

        final_model = finalize_model(best_model)
        output_model = pickle.dumps(final_model)
        b64 = base64.b64encode(output_model).decode()

        href = f'<a href="data:file/output_model;base64,{b64}" download="modelo_entrenado.pkl">Descargar modelo entrenado</a>'
        st.markdown(href, unsafe_allow_html=True)



