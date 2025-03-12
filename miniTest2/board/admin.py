from django.contrib import admin

from board.models import Board, Reply


# Register your models here.
@admin.register(Board)
class BoardAdmin(admin.ModelAdmin):
    pass

@admin.register(Reply)
class ReplyAdmin(admin.ModelAdmin):
    pass