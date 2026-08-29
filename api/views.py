from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse

from django.core.files.base import ContentFile
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db import transaction
from django.utils import timezone

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization

from celery import shared_task
from celery.result import AsyncResult

from .models import TrainingRun

from .utils import retrain_model_with_data, predict_exoplanets, features, active_model, default_model

import os
import pandas as pd
import tempfile
import io
import json
import uuid
import numpy as np

import time

MAX_EPOCHS = 50
MAX_LAYERS = 20
MAX_UNITS = 256
MAX_BATCH_SIZE = 128
MIN_DATASET_SIZE = 100
MAX_DATASET_SIZE = 10000


def classification(request):
	context = {}

	if request.method == 'POST' and request.FILES.get('datafile'):
		try:
			uploaded_file = request.FILES['datafile']

			if not uploaded_file.name.endswith('.csv'):
				raise ValueError("Only CSV files are supported")
			
			df = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))

			if len(df) == 0:
				raise ValueError("File contains no data")
			
			required_cols = features
			missing = [col for col in required_cols if col not in df.columns]
			if missing:
				raise ValueError(f"Missing columns: {', '.join(missing)}")

			results_df = predict_exoplanets(df, default_model)
			context['result'] = results_df.to_html(classes='table table-striped')

		except Exception as e:
			context['error'] = f"Error processing file: {str(e)}"

	return render(request, 'api/classify.html',context)

@shared_task(bind=True,max_retries=3,soft_max_limit=3600,time_limit=7200)
def classification_task(self,file_path,run_id):
	try:
		df = pd.read_csv(file_path,low_memory=False)

		if len(df) == 0:
			raise ValidationError("File contains no data")
		
		required_cols = features
		missing = [col for col in required_cols if col not in df.columns]
		if missing:
			raise ValidationError(f"Missing columns: {', '.join(missing)}")

		training_run = TrainingRun.objects.get(id=run_id)
		model_path= training_run.model_file.path
		model = load_model_from_file(model_path)

		#if len(df) == 0:
		#	raise ValueError("File contains no data")

		results_df = predict_exoplanets(df, model)

		return {
			'status': 'success',
			'result': results_df.to_html(classes='table table-striped')
		}
	except Exception as exc:
		self.retry(exc=exc,countdown=60)

def classify(request, run_id):
	try:
		uploaded_file = request.FILES.get('datafile')

		if not uploaded_file.name.endswith(".csv"):
			raise ValueError("Only CSV files are supported")
		
		with tempfile.NamedTemporaryFile(delete=False,suffix='.csv') as tmp:
			for chunk in uploaded_file.chunks(chunk_size=1024*1024):
				tmp.write(chunk)
			file_path = tmp.name

		#training_run = TrainingRun.objects.get(id=run_id)
		#model_path = training_run.model_file.path
		#model = load_model_from_file(model_path)

		#file_path = io.StringIO(uploaded_file.read().decode('utf-8'))

		#results_df = predict_exoplanets(df, active_model)

		task = classification_task.delay(file_path, run_id)

		return JsonResponse({'task_id': task.id, 'status': 'started'})
	except Exception as exc:
		return JsonResponse({'error': f"Error processing file: {str(exc)}"},status=400)
	#finally:
	
def classification_status(request, task_id):
	try:
		task = AsyncResult(task_id)
		response = {'status': task.status}

		if task.ready():
			if task.successful():
				result = task.result
				response['result'] = {
					'result': result["result"]
				}
			else:
				response['error'] = str(task.result) if task.result else 'Classification failed.'

		return JsonResponse(response)
	except Exception as exc:
		return JsonResponse({'error': str(exc) },status=400)


def sandbox(request):
	training_runs = TrainingRun.objects.all().order_by('-created_at')[:20]
	model_options = [
		{'id': str(run.id), 'version': run.version, 'created_at': run.created_at.strftime('%Y-%m-%d')}
		for run in training_runs
	]

	context = {
		'training_runs': training_runs,
		'model_options_json': json.dumps(model_options),
		'max_epochs': MAX_EPOCHS,
		'max_layers': MAX_LAYERS,
		'max_units': MAX_UNITS,
		'max_batch_size': MAX_BATCH_SIZE
	}
  
	return render(request, 'api/sandbox.html', context)



def constructor(request):
	#data = json.loads(request.POST.get('data'))

	#model_id = data.get('model_id')
	
	#model = TrainingRun.objects.get(id=id)

	context = {
	#	'model': model
	}

	return render(request, 'api/constructor4.html', context)

def constructor_nn(request):
	context = {

	}
	
	return render(request, 'mlapi/')



@shared_task(bind=True, max_retries=3, soft_max_limit=3600, time_limit=7200)
def train_model_task(self,file_path,hyperparams,architecture=None):
	try:
		df = pd.read_csv(file_path, low_memory=False)

		if len(df) < MIN_DATASET_SIZE:
			raise ValueError(f"Dataset too small ({len(df)} rows). Minimum {MIN_DATASET_SIZE} rows required.")

		history,metrics,model,monitor = retrain_model_with_data(
			df, hyperparams, architecture, self, max_epochs=MAX_EPOCHS
		)
		
		model_file = tempfile.NamedTemporaryFile(delete=False,suffix='.h5')

		training_run = save_model(model,model_file.name,hyperparams,metrics,history.history,df,monitor.start_time)

		return {
			'status': 'success',
			'history': history.history,
			'metrics': metrics,
			'model_size': training_run.model_file.size,
			'dataset_size': len(df),
			'training_time': training_run.training_time,
			'current': training_run.id,
			'version': training_run.version
		}
	except Exception as exc:
		raise self.retry(exc=exc,countdown=60)
	#finally:
		#if not "combined" in file_path:
		#	if os.path.exists(file_path):
		#		os.unlink(file_path)

		#if architecture:
		#	model = build_model_from_architecture(architecture)
		#else:
		#	model = Sequential([
		#		Dense(hyperparams)
		#	])

def start_training(request):
	data = json.loads(request.POST.get('data'))

	mode = data.get('mode', 'simple')

	try:
		hyperparams = {
			'learning_rate': float(data.get('learning_rate', 0.001)),
			'batch_size': min(int(data.get('batch_size', 32)), MAX_BATCH_SIZE),
			'epochs': min(int(data.get('epochs', 30)), MAX_EPOCHS),
			'optimizer': data.get('optimizer', 'adam'),

			'beta_1': float(data.get('beta_1', 0.9)),
			'beta_2': float(data.get('beta_2',0.999)),

			'momentum': float(data.get('momentum',0.9)),
			'nesterov': data.get('nesterov', False),

			'use_reg': data.get('use_reg',False),
			'l2_reg': float(data.get('l2_reg', 0.001)),
			'use_batchnorm': data.get('use_batchnorm',False)
		}

		architecture = None

		if mode == 'simple':
			complexity  = data.get('model_complexity', 'medium')

			complexity_map = {
				'small': 32,
				'medium': 64,
				'large': 128,
				'xlarge': min(256, MAX_UNITS)
			}
			units = complexity_map.get(complexity, 64)

			architecture = [
				{'type': 'input', 'units': 8},
				{'type': 'dense', 'units': units, 'activation': 'relu'},
				{'type': 'dense', 'units': max(units//2,8), 'activation': 'relu'},
				{'type': 'output', 'units': 3, 'activation': 'softmax'}
			]
		elif mode == 'advanced':
			print("here3")

			architecture = data.get('architecture',[])#json.loads(data.get('architecture', '[]'))

			if len(architecture) > MAX_LAYERS:
				raise ValidationError(f"Maximum {MAX_LAYERS} allowed.")
			
			for layer in architecture:
				if layer['type'] == 'dense' and layer['units'] > MAX_UNITS:
					layer['units'] = MAX_UNITS

			print("here4")

			if architecture[-1]['type'] != 'output':
				architecture.append({'type': 'output', 'units': 3, 'activation': 'softmax'})

		print("here2")

		uploaded_file = request.FILES.get('training_data')
		print(uploaded_file)
		if not uploaded_file:
			file_path = settings.DEFAULT_DATASET_PATH
		else:
			with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
				for chunk in uploaded_file.chunks(chunk_size=1024*1024):
					tmp.write(chunk)
				file_path = tmp.name

		task = train_model_task.delay(file_path,hyperparams,architecture)

		return JsonResponse({'task_id': task.id, 'status': 'started'})
	except Exception as exc:
		return JsonResponse({'error': str(exc)}, status=400)


def training_status(request, task_id):
	try:
		task = AsyncResult(task_id)
		response = {'status': task.status}

		if task.ready():
			if task.successful():
				result = task.result
				response['result'] = {
					'metrics': result['metrics'],
					'history': result['history'],
					'dataset_size': result['dataset_size'],
					'training_time': result['training_time'],
					'version': result['version'],
					'current': result['current']
				}
			else:
				response['error'] = str(task.result) if task.result else 'Training failed.'
		elif task.status == 'PROGRESS':
			response['progress'] = task.info.get('progress', {})

		return JsonResponse(response)

	except Exception as exc:
		return JsonResponse({'error': str(exc)}, status=400)
		

@require_POST
def save_trained_model(request):
	try:
		task_id = request.POST.get('task_id')
		description = request.POST.get('description', '')

		task = AsyncResult(task_id)
		if not task.ready() or not task.successful():
			return JsonResponse({'error': 'Training not complete'}, status=400)
		
		result = task.result

		with transaction.atomic():
			training_run = TrainingRun(
				version=str(uuid.uuid4())[:8],
				description=description,
				hyperparameters=result.get('hyperparams', {}),
				metrics=result['metrics'],
				training_history=result['history'],
				dataset_size=result['dataset_size'],
				training_time=result['training_time']
			)
			
			model_filename = f"model_v{training_run.version}.h5"
			with open(result['model_file'], 'rb') as f:
				training_run.model_file.save(model_filename, ContentFile(f.read()))

			training_run.save()

		if os.path.exists(result['model_file']):
			os.unlink(result['model_file'])

		return JsonResponse({
			'success': True,
			'version': training_run.version,
			'run_id': training_run.id
		})
	except Exception as exc:
		return JsonResponse({'error': str(exc)},status=400)

@require_POST
def load_model(request, run_id):
	try:
		training_run = TrainingRun.objects.get(id=run_id)

		model_path = training_run.model_file.path
		model = load_model_from_file(model_path)

		global active_model
		active_model = model

		return JsonResponse({
			'success': True,
			'version': training_run.version,
			'current': training_run.id,
			'architecture': json.loads(model.to_json()),
			'message': f"Model v{training_run.version} loaded successfully"
		})
	except Exception as exc:
		return JsonResponse({'error': str(exc)},status=400)
	
#def load_constructor(request, id):
#	try:
#		constructor = Constructor.objects.get(id=run_id)
#
#		return JsonResponse({
#
#		})
#	except Exception as exc:
#		return JsonResponse({'error': str(exc)},status=400)
	
def save_model(model,filename,hyperparameters_,metrics_,history,df,start_time):
	model.save(filename)

	training_run = TrainingRun(
		version=str(uuid.uuid4())[:8],
		hyperparameters=hyperparameters_,
		metrics={
			'accuracy': metrics_["accuracy"],
			'loss': metrics_["loss"],
			'precision': 0.0,
			'recall': 0.0, 
		},
		training_history = history,
		dataset_size=len(df),
		training_time=time.time() - start_time
	)

	with open(filename, 'rb') as f:
		training_run.model_file.save(f'v{training_run.version}.h5', ContentFile(f.read()))
	
	training_run.save()

	return training_run
	#tf.keras.models.load_model(filename)

def load_model_from_file(model_path):
	model = tf.keras.models.load_model(model_path)
	#return
	
	return model

def delete_model(request,run_id):
	try:
		print(run_id)
		training_run = TrainingRun.objects.get(id=run_id)

		training_run.model_file.delete()
		training_run.delete()

		return JsonResponse({'success': True})
	except TrainingRun.DoesNotExist:
		return JsonResponse({'error': 'Model not found'}, status=404)

def compare_models(request):
	run1_id = request.GET.get('run1')
	run2_id = request.GET.get('run2')

	if not run1_id or not run2_id:
		return HttpResponseBadRequest("Missing run IDs")
	
	try:
		run1 = TrainingRun.objects.get(id=run1_id)
		run2 = TrainingRun.objects.get(id=run2_id)

		comparison_data = {
			'run1': {
				'id': run1.id,
				'version': run1.version,
				'metrics': run1.metrics,
				'history': run1.training_history,
				'training_time': run1.training_time,
				'model_size': run1.model_file.size if run1.model_file else 0
			},
			'run2': {
				'id': run2.id,
				'version': run2.version,
				'metrics': run2.metrics,
				'history': run2.training_history,
				'training_time': run2.training_time,
				'model_size': run2.model_file.size if run2.model_file else 0
			},
			'metrics': {
				'accuracy': {
					'run1': run1.metrics["accuracy"],
					'run2': run2.metrics["accuracy"],
				},
				'loss': {
					'run1': run1.metrics["loss"],
					'run2': run2.metrics["loss"]
				}
				#'run1': run1.metrics,
				#'run2': run2.metrics
			}
		}

		return JsonResponse(comparison_data)
	except TrainingRun.DoesNotExist:
		return JsonResponse({'error': "One or more models not found"}, status=404)
	
def list_models(request):
	training_runs = TrainingRun.objects.all().order_by('-created_at')[:20]
	models_data = [
		{
			'id': run.id,
			'version': run.version,
			'created_at': run.created_at.strftime('% Y - % m - % d'),
			'accuracy': run.metrics.get('accuracy', 0),
			'loss': run.metrics.get('loss', 0),
			'dataset_size': run.dataset_size
		}
		for run in training_runs
	]
	return JsonResponse({'models': models_data})

def model_details(request, run_id):
	try:
		run = TrainingRun.objects.get(id=run_id)
		return JsonResponse({
			'id': run.id,
			'version': run.version,
			'created_at': run.created_at.strftime('%Y-%m-%d %H:%M:%S'),
			'description': run.description,
			'hyperparameters': run.hyperparameters,
			'metrics': run.metrics,
			'dataset_size': run.dataset_size,
			'training_time': run.training_time,
			'model_size': run.model_file.size if run.model_file else 0
		})
	except TrainingRun.DoesNotExist:
		return JsonResponse({'error': 'Model not found'}, status=400)

def build_model_from_architecture222(architecture):
	model = Sequential()
    
	# Add layers based on architecture
	for i, layer in enumerate(architecture):
		if layer['type'] == 'input':
			# Input layer is implicit
			continue
            
		elif layer['type'] == 'dense':
			if i == 1:  # First layer after input
				model.add(Dense(layer['units'], activation=layer['activation'], input_shape=(8,)))
			else:
				model.add(Dense(layer['units'], activation=layer['activation']))
                
		elif layer['type'] == 'dropout':
			model.add(Dropout(layer['rate']))
            
		elif layer['type'] == 'batchnorm':
			model.add(BatchNormalization())
            
		elif layer['type'] == 'output':
			model.add(Dense(layer['units'], activation=layer['activation']))
    
	return model

def validate_architecture(architecture):
	if not architecture:
		return False
	
	has_input = any(layer['type'] == 'input' for layer in architecture)
	has_output = any(layer['type'] == 'output' for layer in architecture)

	if not has_input or not has_output:
		return False
	
	for layer in architecture:
		if layer['type']  == 'dense' and layer['units'] > MAX_UNITS:
			return False
		
	return True