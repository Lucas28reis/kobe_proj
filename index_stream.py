import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import mlflow
from sklearn.metrics import *
from mlflow import MlflowClient
from sklearn import linear_model, preprocessing, metrics, model_selection

# URI de Tracking do MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db") 

# Identifique o experimento e run
experiment_name = "Monitoramento"
run_name_dev = "modelo_kobe"
run_name_apk = "PipelineAplicacao"

# Baixar os artefatos diretamente para o sistema de arquivos local
client = mlflow.tracking.MlflowClient()

# Buscar a última versão do modelo
latest_version_info = client.get_latest_versions(run_name_dev, stages=["Staging", "Production"])
latest_version = latest_version_info[-1]  # Pega a última versão na lista, assumindo que a lista esteja ordenada

# Recuperar o run_id da última versão
run_id_mod = latest_version.run_id
print(f"O run_id da última versão do modelo '{run_name_dev}' é: {run_id_mod}")

# run_id_mod = "d7910a20f26e4e9ba7f470e9db3d40b5"  # ID do run específico do modelo
run_id_apk = "8e37ce90cc2b4f1d9fad82ba4b4fe246"  # ID do run específico da aplicação
name_apk = "prediction_proba.parquet"
name_mod = "tune_test.parquet"
target = 'shot_made_flag'


local_dir = "download"  # Diretório de downloads
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# Função para baixar os arquivos
def download_artifact(run_id, artifact_path, local_dir):
    try:
        client.download_artifacts(run_id=run_id, path=artifact_path, dst_path=local_dir)
        print(f"Downloaded {artifact_path} from run {run_id} to {local_dir}")

    except Exception as e:
        print(f"Failed to download {artifact_path} from run {run_id}: {e}")

download_artifact(run_id_mod, name_mod, local_dir)
download_artifact(run_id_apk, name_apk, local_dir)

df_prod = pd.read_parquet(f"{local_dir}/prediction_proba.parquet")
df_dev = pd.read_parquet(f"{local_dir}/tune_test.parquet")


# PAINEL COM AS PREVISÕES HISTÓRICAS
view_image = "https://image.api.playstation.com/vulcan/ap/rnd/202307/2809/a9fe3100a67c5d90e6cc5d24ebab09b8305468493192deac.jpg"
st.header("🏀 Black Mamba")
st.write("O propósito desta pesquisa é utilizar métodos de inteligência artificial para determinar se um lançamento resultará em pontos")
st.image(view_image, width=700)


st.subheader("Tabela de Métricas do Modelo de Desenvolvimento", divider='rainbow')
tab1, tab2, tab3 = st.tabs(["DataFrame", "Tabela Report",  "Matriz de Confusão"])
with tab1:
    st.write(df_dev.head())

with tab2:
    report_2 = classification_report(df_dev[target], df_dev['prediction_label'], output_dict=True)
    report_dev = pd.DataFrame(report_2).transpose()
    st.table(report_dev)
    st.write(f"Tamanho da Base de Desenvolvimento: {df_dev.shape[0]}")

with tab3:
    # Plot da Matriz de Confusão
    actual_label = 'shot_made_flag'
    y_true_dev = df_dev[actual_label]
    y_pred_dev = df_dev['prediction_label']
    conf_mat = confusion_matrix(y_true_dev, y_pred_dev)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax)
    ax.set(title='Matriz de Confusão - Produção', xlabel='Previsto', ylabel='Real')
    st.pyplot(fig)



st.subheader("Tabela de Métricas do Modelo de Produção", divider='rainbow')
tab1, tab2, tab3, tab4 = st.tabs(["DataFrame", "Plots", "Tabela Report",  "Matriz de Confusão"])

with tab1:
    st.write(df_prod.head())


with tab2:    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x=df_prod[target].value_counts().index,
        height=df_prod[target].value_counts().values)
    ax.set_title('Histograma do Target Teste')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Não', 'Sim'])

# Exibindo o gráfico no Streamlit
    st.pyplot(fig)

with tab3:

    report = classification_report(df_prod[target], df_prod['operation_label'], output_dict=True)
    report_prov = pd.DataFrame(report).transpose()
    st.table(report_prov)
    st.write(f"Tamanho da Base de Produção: {df_prod.shape[0]}")

with tab4:
    # Plot da Matriz de Confusão
    actual_label = 'shot_made_flag'
    y_true_prod = df_prod[actual_label]
    y_pred_prod = df_prod['operation_label']
    conf_mat = confusion_matrix(y_true_prod, y_pred_prod)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_mat, annot=True, fmt='d', ax=ax)
    ax.set(title='Matriz de Confusão - Produção', xlabel='Previsto', ylabel='Real')
    st.pyplot(fig)




st.header("Probabilidade de Acerto do Arremesso", divider='rainbow')

# Plot da Distribuição das Probabilidades de Previsão
fig, ax = plt.subplots(figsize=(5, 3))
sns.histplot(df_dev['prediction_score_1'], kde=True, label='Teste', ax=ax)
sns.histplot(df_prod['predict_proba'], kde=True, label='Produção', ax=ax)
ax.set(title='Saída do Modelo', xlabel='Probabilidade de Acerto do Arremesso', ylabel='Pontuação Estimada')
plt.legend()
st.pyplot(fig)


# # Curva ROC e AUC
y_score_prod = df_prod['operation_label']
y_score_dev = df_dev['prediction_score_1']


# Curva ROC para o modelo de Produção
fpr_prod, tpr_prod, thr_prod = roc_curve(y_true_prod, y_score_prod)
auc_prod = auc(fpr_prod, tpr_prod)

# Curva ROC para o modelo de Desenvolvimento
fpr_dev, tpr_dev, thr_dev = roc_curve(y_true_dev, y_score_dev)
auc_dev = auc(fpr_dev, tpr_dev)

# Plotando as curvas ROC
fig, ax = plt.subplots(figsize=(5, 3))
plt.plot(fpr_prod, tpr_prod, label=f'Produção (AUC = {auc_prod:.2f})')
plt.plot(fpr_dev, tpr_dev, label=f'Desenvolviemnto (AUC = {auc_dev:.2f})')

# Adicionando elementos gráficos
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC para Comparação de Modelos')
plt.legend(loc="lower right")
st.pyplot(fig)
