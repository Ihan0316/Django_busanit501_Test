from django.db import models

# Create your models here.
class Post(models.Model):
    title = models.CharField("포스트 제목", max_length=100)
    content = models.TextField("포스트 내용")
    # 사용자가 업로드한 이미지 추가하는 필드
    # /memdia/post/
    thumbnail = models.ImageField("썸네일 이미지",
                                  upload_to="post", blank=True)

    def __str__(self):
        return self.title
# 댓글 모델도 같이 정의.
class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    content = models.TextField("댓글 내용")

    def __str__(self):
        return f"{self.post.title}의 댓글 ID:{self.id}"


