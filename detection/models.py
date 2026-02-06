from django.db import models

# Create your models here.

class NewsQuery(models.Model):
    text_input = models.TextField(blank=True, null=True)
    url_input = models.URLField(blank=True, null=True)
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.url_input or self.text_input[:50]} - {self.prediction}"
