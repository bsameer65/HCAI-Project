from django.test import SimpleTestCase
from django.urls import reverse


class HomeRoutingTests(SimpleTestCase):
    def test_root_redirects_to_home(self):
        response = self.client.get("/")

        self.assertRedirects(
            response,
            reverse("home:index"),
            status_code=302,
            fetch_redirect_response=False,
        )
