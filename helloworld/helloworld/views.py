from django.http import HttpResponse
from django.shortcuts import render


def main(request):
    # return HttpResponse("안녕하세요. 장고 웹 프레임워크~~ Hello World")
    return render(request, "main.html")

def lunch_list(request):
    # return HttpResponse("lunch_list : 점심 메뉴입니다.")
    return render(request, "lunch_list.html")

def intro(request):
    return HttpResponse("반갑습니다. 이상용입니다.")



