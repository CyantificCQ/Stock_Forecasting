from django.urls import path
from gold_price import views 

urlpatterns = [
    path("", views.prediction, name="prediction"),
    path("result", views.formInfo, name="result")
    # path("", views.Welcome, name="Welcome"),
    # path("user", views.User, name= "User"),
]

