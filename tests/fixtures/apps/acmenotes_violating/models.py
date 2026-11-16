from django.db import models


class Profile(models.Model):
    email = models.EmailField()
    display_name = models.CharField(max_length=80)
    phone = models.CharField(max_length=15, blank=True)
    ip_address = models.GenericIPAddressField(null=True)
    aadhaar_number = models.CharField(max_length=12)   # never disclosed in the policy
    date_of_birth = models.DateField()                  # never disclosed in the policy