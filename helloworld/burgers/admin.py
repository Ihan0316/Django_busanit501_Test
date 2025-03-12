from django.contrib import admin
from django.contrib.auth.models import User

from burgers.models import Burger


# Register your models here.
@admin.register(Burger)
class BurgerAdmin(admin.ModelAdmin):
    pass
