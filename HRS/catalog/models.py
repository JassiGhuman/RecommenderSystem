from django.db import models
# model for passing values from dataset to the xgboost algorithm


class RecordTable(models.Model):
    site_name = models.IntegerField()
    user_location_region = models.IntegerField()
    is_package = models.IntegerField()
    srch_adults_cnt = models.IntegerField()
    srch_children_cnt = models.IntegerField()
    srch_destination_id = models.IntegerField()
    hotel_market = models.IntegerField()
    hotel_country = models.IntegerField()
    hotel_cluster = models.IntegerField()

    def __str__(self):
        return self.site_name






