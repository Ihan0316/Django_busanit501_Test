from django.shortcuts import render

from board.models import Board


# Create your views here.

def board_list(request):
    boards = Board.objects.all()
    context = {
        'boards': boards,
    }
    return render(request,"board_list.html",context)