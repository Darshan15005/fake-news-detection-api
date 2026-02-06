from django.urls import path
from .api_views import detect_news, health_check

urlpatterns = [
    path("predict/", detect_news),
    path("health/", health_check)
]
