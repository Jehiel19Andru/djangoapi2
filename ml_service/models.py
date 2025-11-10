from mongoengine import Document, IntField, FloatField, DateTimeField
import datetime

class TrainingResult(Document):
    # Añadido el nuevo campo de interés
    num_samples = IntField(required=True) 
    
    n_estimators = IntField(required=True)
    n_features_selected = IntField(required=True)
    f1_score = FloatField(required=True)
    timestamp = DateTimeField(default=datetime.datetime.utcnow)

    meta = {
        'collection': 'training_results'
    }