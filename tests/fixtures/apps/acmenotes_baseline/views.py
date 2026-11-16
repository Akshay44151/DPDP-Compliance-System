from django.core.mail import send_mail

from .models import Profile


def register(request):
    consent = request.POST.get("marketing_consent") == "on"
    profile = Profile.objects.create(
        email=request.POST["email"],
        display_name=request.POST["display_name"],
        phone=request.POST.get("phone", ""),
        ip_address=request.META.get("REMOTE_ADDR"),
        marketing_consent=consent,
    )
    if profile.marketing_consent:
        send_mail("Product updates", "What's new", "noreply@acmenotes.example", [profile.email])