from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from acmenotes_baseline.models import Profile


class Command(BaseCommand):
    help = "Delete accounts inactive for two years."

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(days=730)
        Profile.objects.filter(last_login__lt=cutoff).delete()