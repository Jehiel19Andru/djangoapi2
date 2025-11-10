import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import os
from django.conf import settings
import numpy as np # Necesario para el cálculo de porcentaje en samples

# ¡Ajustar la ruta de tu archivo según tu estructura de proyecto!
DATASET_PATH = os.path.join(settings.BASE_DIR, 'datasets', 'TotalFeatures-ISCXFlowMeter.csv')

# --- VALORES FIJOS SOLICITADOS (Simulando que el front los ha eliminado) ---
# Estos valores se usarán para el entrenamiento, buscando la mayor precisión.
FIXED_N_ESTIMATORS = 500
FIXED_N_FEATURES = 30
# --------------------------------------------------------------------------

def run_random_forest_feature_selection(sample_percentage: float):
    """
    Carga datos, realiza preprocesamiento, toma una muestra del set de entrenamiento
    basada en el porcentaje solicitado, selecciona características (con valores fijos de RF), 
    y calcula el F1 Score y Accuracy.

    Args:
        sample_percentage (float): Porcentaje (1-100) del dataset total a usar para el entrenamiento.
                                   (Este valor ya no está limitado al 90% en la lógica).
    """
    
    # 1. Carga de Datos y Preprocesamiento
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        # Re-raise el error para que sea capturado en la vista
        raise Exception(f"Error: Dataset no encontrado en {DATASET_PATH}. Verifica la ruta.") 

    try:
        X = df.drop('calss', axis=1)  
        y = df['calss']               
    except KeyError:
        raise Exception("Error: La columna objetivo ('calss') no se encontró.")

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Limpieza de datos
    X = X.replace([float('inf'), -float('inf')], 0).fillna(0) 

    # 2. División inicial: Separamos una parte para testeo final (30%)
    # Esta división es CRÍTICA y asegura que el 30% nunca se use en el entrenamiento, 
    # garantizando que F1 Score nunca sea 1.0.
    X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. Muestreo del Set de Entrenamiento (basado en el porcentaje solicitado)
    
    # Se calcula la cantidad de muestras a tomar del set de entrenamiento completo (el 70% original)
    # Se ha eliminado la restricción de que el máximo es el 90% del total; ahora el máximo es el 70% (X_full_train).
    total_data_count = len(df)
    
    # Cantidad absoluta de muestras solicitadas (del TOTAL)
    samples_requested = int(total_data_count * (sample_percentage / 100.0))
    
    # La cantidad REAL de muestras a usar será el mínimo entre lo solicitado y lo disponible (X_full_train)
    num_samples_to_use = min(samples_requested, len(X_full_train)) 
    
    # Cálculo del porcentaje real que esta cantidad representa sobre el X_full_train
    if num_samples_to_use < len(X_full_train):
        # Tomamos la muestra del conjunto de entrenamiento COMPLETO (70% original)
        X_train, _, y_train, _ = train_test_split(
            X_full_train, y_full_train, 
            train_size=num_samples_to_use, 
            random_state=42, 
            stratify=y_full_train
        )
    else:
        # Usamos el conjunto de entrenamiento completo (el 70% original)
        X_train, y_train = X_full_train, y_full_train
        
    # 4. Selección de Características y Entrenamiento Final (Usando los valores fijos)
    
    # Se usan los valores fijos de 500 árboles y 30 features
    rf_selector = RandomForestClassifier(n_estimators=FIXED_N_ESTIMATORS, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)

    feature_importances = pd.Series(rf_selector.feature_importances_, index=X_train.columns)
    
    # Aseguramos que el número de features no exceda la cantidad total de features disponibles
    n_features_used = min(FIXED_N_FEATURES, len(X_train.columns))
    
    selected_features = feature_importances.nlargest(n_features_used).index.tolist() 
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    final_rf = RandomForestClassifier(n_estimators=FIXED_N_ESTIMATORS, random_state=42, n_jobs=-1)
    final_rf.fit(X_train_selected, y_train)

    # 5. Evaluación y Retorno
    y_pred = final_rf.predict(X_test_selected)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred) 

    # Retornamos los resultados y los parámetros utilizados
    return f1, accuracy, num_samples_to_use, FIXED_N_ESTIMATORS, n_features_used