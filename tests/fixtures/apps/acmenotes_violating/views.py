import logging

import requests
from django.core.mail import send_mail

from .models import Profile

logger = logging.getLogger(__name__)


def register(request):
    email = request.POST["email"]
    logger.info("New registration from %s", email)
    Profile.objects.create(
        email=email,
        display_name=request.POST["display_name"],
        phone=request.POST.get("phone", ""),
        aadhaar_number=request.POST["aadhaar"],
        date_of_birth=request.POST["dob"],
        ip_address=request.META.get("REMOTE_ADDR"),
    )
    send_mail("Weekly offers", "Check out our deals", "noreply@acmenotes.example", [email])
    requests.post("https://ads.example.com/track", data={"email": email})
