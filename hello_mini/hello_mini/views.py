from django.http import HttpResponse
from django.shortcuts import render


def main(request):
    return HttpResponse("Hello, world!")


def test(request):
    return render(request,"test.html")