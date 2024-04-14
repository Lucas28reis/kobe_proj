import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from sklearn import metrics
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import *

col_2 = ['lat','lon', 'minutes_remaining', 'period','shot_distance',
'playoffs']
col = ['lat','lon', 'minutes_remaining', 'period','shot_distance',
'playoffs', 'shot_made_flag']

model = 'modelo_kobe'
alias = 'staging'
target = 'shot_made_flag'
threshold = 0.35
min_eff_alarm = 0.1
pred_holdout = 0.55


mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Monitoramento'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id=experiment.experiment_id

client = mlflow.MlflowClient()

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):
    model_uri = f"models:/modelo_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    data_prod = pd.read_parquet('data/raw/dataset_kobe_prod.parquet')
    data_prod = data_prod[col].dropna()

    Y = loaded_model.predict_proba(data_prod[col_2])
    data_prod['predict_proba'] = Y[:, 1]  # Adiciona a probabilidade de uma das classes, se binário
    data_prod['operation_label'] = (data_prod['predict_proba'] >= threshold).astype(int)


    Z = loaded_model.predict(data_prod[col_2])
    data_prod['predict_score'] = Z  # Adiciona predições de classe


    print(metrics.classification_report(data_prod[target], data_prod['operation_label']))

    # # LOG DE METRICAS GLOBAIS
    # cm = metrics.confusion_matrix(data_prod[target], data_prod['operation_label'])
    # specificity = cm[0,0] / cm.sum(axis=1)[0]
    # sensibility = cm[1,1] / cm.sum(axis=1)[1]
    # precision   = cm[1,1] / cm.sum(axis=0)[1]

    def alarm(data_monitoring, min_eff_alarm, metric_t=threshold):
        cm = metrics.confusion_matrix(data_monitoring[target], data_monitoring['operation_label'])
        specificity_m = cm[0,0] / (cm.sum(axis=1)[0] if cm.sum(axis=1)[0] else 1)
        sensibility_m = cm[1,1] / (cm.sum(axis=1)[1] if cm.sum(axis=1)[1] else 1)
        precision_m = cm[1,1] / (cm.sum(axis=0)[1] if cm.sum(axis=0)[1] else 1)

        retrain = False
        for name, metric_m in zip(['especificidade', 'sensibilidade', 'precisao'], [specificity_m, sensibility_m, precision_m]):
            print(f'\t=> {name} de teste {metric_t} e de controle {metric_m}')
            if (metric_t - metric_m) / metric_t > min_eff_alarm:
                print(f'\t=> MODELO OPERANDO FORA DO ESPERADO')
                retrain = True
            else:
                print(f'\t=> MODELO OPERANDO DENTRO DO ESPERADO')
                
        return (retrain, [specificity_m, sensibility_m, precision_m])

    (retrain, [specificity_m, sensibility_m, precision_m]) = alarm(data_prod, 0.15)  # Exemplo de threshold de alarme ajustado

    if retrain:
        print('==> RETREINAMENTO NECESSARIO')
    else:
        print('==> RETREINAMENTO NAO NECESSARIO')

    # LOG DE PARAMETROS DO MODELO
    mlflow.log_param("min_eff_alarm", min_eff_alarm)

    # LOG DE METRICAS GLOBAIS
    mlflow.log_metric("Alarme Retreino", float(retrain))
    mlflow.log_metric("Especificidade Controle", specificity_m)
    mlflow.log_metric("Sensibilidade Controle", sensibility_m)
    mlflow.log_metric("Precisao Controle", precision_m)
    

    data_prod.to_parquet('data/processed/prediction_proba.parquet')
    mlflow.log_artifact('data/processed/prediction_proba.parquet')
    mlflow.log_metric('log_loss',log_loss(data_prod.shot_made_flag, data_prod.predict_proba))
    mlflow.log_metric('f1', f1_score(data_prod.shot_made_flag, data_prod.predict_score))
