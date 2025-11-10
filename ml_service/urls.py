from django.urls import path
from .views import FeatureSelectionAPI

urlpatterns = [
    # Esta es la URL que tu JavaScript está llamando: /api/ml/train/
    path('train/', FeatureSelectionAPI.as_view(), name='train_model'),
]