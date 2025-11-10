from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView # <--- Correcto

urlpatterns = [
    # SOLUCIÓN: La URL raíz ('/') ahora carga el template index.html
    path('', TemplateView.as_view(template_name="index.html"), name='home'),
    
    # Rutas existentes:
    path('admin/', admin.site.urls),
    # Esta línea ahora funcionará porque creamos 'ml_service.urls'
    path('api/ml/', include('ml_service.urls')), 
]