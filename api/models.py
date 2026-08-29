from django.db import models

import uuid

class TrainingRun(models.Model):
	id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
	created_at = models.DateTimeField(auto_now_add=True)
	version = models.CharField(max_length=20, unique=True)
	description = models.TextField(blank=True)
	hyperparameters = models.JSONField()
	metrics = models.JSONField()
	model_file = models.FileField(upload_to='models/')
	training_history = models.JSONField()
	dataset_size = models.IntegerField()
	training_time = models.FloatField() # in seconds

	model_size = models.IntegerField(default=0)

	def save(self, *args, **kwargs):
		if self.model_file:
			self.model_size = self.model_file.size
			super().save(*args,**kwargs)
    
	def __str__(self):
		return f"Model v{self.version} ({self.created_at})" 