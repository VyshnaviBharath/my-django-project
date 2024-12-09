from django.urls import path
from .views import home, login, signup, signup_success, login_success, classify_image, logout_view

urlpatterns = [
    path('', home, name='home'),
    path('login/', login, name='login'),
    path('signup/', signup, name='signup'),
    path('signup/success/', signup_success, name='signup_success'),
    path('login/success/', login_success, name='login_success'),
    path('logout/', logout_view, name='logout'),
    path('classify_image/', classify_image, name='classify_image'),  # Add this line
]