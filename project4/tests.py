from django.test import SimpleTestCase
from django.urls import reverse


class Project4PageTests(SimpleTestCase):
    def test_landing_page_loads(self):
        response = self.client.get(reverse("project4:index"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Preference Elicitation")
        self.assertContains(response, "Start the study")

    def test_study_intro_page_loads(self):
        response = self.client.get(reverse("project4:study"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Before you begin")

