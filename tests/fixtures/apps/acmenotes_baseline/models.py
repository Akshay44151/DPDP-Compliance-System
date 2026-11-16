from django.db import models


class Profile(models.Model):
    email = models.EmailField()
    display_name = models.CharField(max_length=80)
    phone = models.CharField(max_length=15, blank=True)
    ip_address = models.GenericIPAddressField(null=True)
    device_id = models.CharField(max_length=200, blank=True)
    marketing_consent = models.BooleanField(default=False)
    scheduled_deletion_at = models.DateTimeField(null=True)