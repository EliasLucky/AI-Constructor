"""
URL configuration for aiconstructor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

#from ratelimit.decorators import ratelimit

from homepage import views as home
from about import views as about
from api import views as api

from aiconstructor import settings

urlpatterns = [
    path('', home.index, name="homepage"),
	path('about/', about.index, name="about"),
	
    path('classify/', api.classification, name="classify"),
	path('sandbox/', api.sandbox, name="sandbox"),
	path('constructor/', api.constructor, name="constructor"),

    path('api/train/', api.start_training, name="start_training"),
	path('api/train/status/<str:task_id>/', api.training_status, name='training_status'),
	path('api/predict/<uuid:run_id>/', api.classify, name='predict'),
	path('api/predict/status/<str:task_id>/', api.classification_status, name='classification_status'),
    path('api/models/save/', api.save_trained_model, name='save_model'),

    path('api/models/<uuid:run_id>/load/', api.load_model, name='load_model'),
	path('api/models/<uuid:run_id>/delete/', api.delete_model, name='delete_model'),

    path('api/compare/', api.compare_models, name='compare_models'),
	
    path('api/models/', api.list_models, name='list_models'),
	path('api/models/<uuid:run_id>/', api.model_details, name='model_details'),
]

if settings.DEBUG:
	urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)