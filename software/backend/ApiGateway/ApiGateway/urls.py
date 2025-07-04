"""
URL configuration for ApiGateway project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import re_path
from gateway.proxy.views import OrthancProxyView ,NiftiProxyView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('dicom/', include('gateway.orthanc.urls')),
    path('users/', include('gateway.users.urls')),
    path('inference/', include('gateway.inference.urls')),
    re_path(r'^orthanc/(?P<path>.*)$', OrthancProxyView.as_view()),
    
    # Niftyproxy is
    re_path(r'^nifti/(?P<path>.*)$', NiftiProxyView.as_view()),
]

