from django.db import models


class HumanExpertResponse(models.Model):
    """
    Stores one annotation provided through the interactive human-expert
    extension.
    """

    CLASS_CHOICES = [
        (0, "World"),
        (1, "Sports"),
        (2, "Business"),
        (3, "Sci/Tech"),
    ]

    article_index = models.PositiveIntegerField()

    article_text = models.TextField()

    selected_label = models.IntegerField(
        choices=CLASS_CHOICES,
    )

    true_label = models.IntegerField(
        choices=CLASS_CHOICES,
    )

    is_correct = models.BooleanField()

    query_strategy = models.CharField(
        max_length=100,
    )

    classifier_prediction = models.IntegerField(
        choices=CLASS_CHOICES,
        null=True,
        blank=True,
    )

    classifier_confidence = models.FloatField(
        null=True,
        blank=True,
    )

    classifier_entropy = models.FloatField(
        null=True,
        blank=True,
    )

    session_key = models.CharField(
        max_length=100,
        db_index=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return (
            f"Article {self.article_index} - "
            f"{self.get_selected_label_display()}"
        )