from rest_framework import serializers

class TrainRequestSerializer(serializers.Serializer):
    n_estimators = serializers.IntegerField(min_value=10, max_value=500)
    n_features = serializers.IntegerField(min_value=1, max_value=80)