# apps/recruiter_dashboard/urls.py

from django.urls import path

from apps.recruiter_dashboard.views import DashboardView

app_name = "recruiter_dashboard"

urlpatterns = [
    path("", DashboardView.as_view(), name="index"),
]