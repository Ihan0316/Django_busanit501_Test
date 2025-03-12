from rest_framework import serializers

from blog.models import Post

# 직렬화? , 객체 -> JSON 형태 변경.
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ["id", "title", "content", "created_at"]  # JSON으로 변환할 필드 지정
