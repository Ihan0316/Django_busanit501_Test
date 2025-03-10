from django.http import HttpResponse
from django.shortcuts import render

from burgers.models import Burger


def main(request):
    # return HttpResponse("안녕하세요. 장고 웹 프레임워크~~ Hello World")
    return render(request, "main.html")

def lunch_list(request):
    # return HttpResponse("lunch_list : 점심 메뉴입니다.")
    return render(request, "lunch_list.html")

def intro(request):
    return HttpResponse("반갑습니다. 이상용입니다.")

def burger_list(request):
    burgers = Burger.objects.all()
    print("햄버거 전체 목록: ", burgers)
    # 화면에 데이터 탑재하기.
    context = {
        "burgers": burgers
    }
    # 화면에 데이터 전달하기.
    return render(request, "burger_list.html", context)

def burger_search(request):
    # print(request.GET)
    keyword = request.GET.get("keyword")
    print(keyword)

    # 유효성 체크, keyword 값이 주어진 경우에만 조회하기.
    if keyword is not None:
        # 검색어 이용해서, DB에서 데이터 검색하기.
        burgers = Burger.objects.filter(name__contains=keyword)
    else:
        # 검색어가 없는 경우.
        burgers = Burger.objects.none()
    # 서버 -> 화면으로 전달
    context = {
        "burgers": burgers
    }
    print("조회된 내용 : ",burgers)
    return render(request, "burger_search.html",context)



