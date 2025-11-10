from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import run_random_forest_feature_selection 
from .models import TrainingResult 

# ¡AJUSTE CRÍTICO! Usa el valor real de tu dataset
TOTAL_SAMPLES = 631955 
# El 90% de los datos totales es el límite absoluto de entrenamiento
MAX_TRAINING_SAMPLES = int(TOTAL_SAMPLES * 0.9) 

class FeatureSelectionAPI(APIView):
    def post(self, request, *args, **kwargs):
        
        try:
            # 1. Obtener los parámetros del request
            sample_percentage = float(request.data.get('sample_percentage', 4.0)) 
            n_estimators = int(request.data.get('n_estimators', 100)) 
            n_features = int(request.data.get('n_features', 10))     

            # 2. CÁLCULO CRÍTICO: Convertir el porcentaje a número de muestras y aplicar el límite
            
            # Cantidad de muestras basada en el porcentaje solicitado
            requested_samples = int((sample_percentage / 100.0) * TOTAL_SAMPLES)
            
            # Aplicar la regla de negocio: Nunca entrenar con más del 90% de los datos totales
            num_samples = min(
                requested_samples,       # Límite por porcentaje del usuario
                MAX_TRAINING_SAMPLES     # Límite de la regla de negocio (90% del total)
            )
            # Asegurar que al menos sea 1 muestra
            num_samples = max(1, num_samples)


            # 3. Llamar a la función de Machine Learning 
            f1_score_result, accuracy_result, used_samples, used_estimators, used_features = run_random_forest_feature_selection(
                num_samples=num_samples, 
                n_estimators=n_estimators,
                n_features=n_features
            ) # ¡AJUSTE! Ahora recibe Accuracy

            # 4. Guardar en MongoDB
            try:
                new_result = TrainingResult(
                    num_samples=used_samples,
                    n_estimators=used_estimators,
                    n_features_selected=used_features,
                    f1_score=f1_score_result,
                    accuracy_score=accuracy_result # ¡AJUSTE! Guardar Accuracy
                )
                new_result.save() 
            except Exception as e:
                print(f"Advertencia: Error al guardar en MongoDB: {e}")

            # 5. Preparar la respuesta JSON
            response_data = {
                "parameters_used": {
                    "sample_percentage": sample_percentage, 
                    "num_samples": used_samples, 
                    "n_estimators": used_estimators,
                    "n_features_selected": used_features 
                },
                "f1_score": f1_score_result,
                "accuracy_score": accuracy_result # ¡AJUSTE! Retornar Accuracy
            }

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"error": f"Error en el procesamiento: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )