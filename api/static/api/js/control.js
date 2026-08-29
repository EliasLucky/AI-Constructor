import $ from '/static/api/js/jquery-module.js'

var accChart;
var lossChart;

var savedCharts;

function updateTrainingCharts(history) {
	if (!accChart && !lossChart) {
		savedCharts = history;

		const lastEpoch = history.accuracy.length-1;
		$('#acc-value').text(history.val_accuracy[lastEpoch].toFixed(4));
		$('#loss-value').text(history.val_loss[lastEpoch].toFixed(4));

		return;
	}

	savedCharts = null;
  
	accChart.data = {
		labels: Array.from({length: history.accuracy.length}, (_, i) => i+1),
		datasets: [
			{ label: 'Training Accuracy', data: history.accuracy, borderColor: 'blue' },
			{ label: 'Validation Accuracy', data: history.val_accuracy, borderColor: 'green' }
		]
	};
	accChart.update();
  
	lossChart.data = {
		labels: Array.from({length: history.loss.length}, (_, i) => i+1),
		datasets: [
			{ label: 'Training Loss', data: history.loss, borderColor: 'red' },
			{ label: 'Validation Loss', data: history.val_loss, borderColor: 'orange' }
		]
	};
	lossChart.update();
  
	const lastEpoch = history.accuracy.length - 1;
	$('#acc-value').text(history.val_accuracy[lastEpoch].toFixed(4));
	$('#loss-value').text(history.val_loss[lastEpoch].toFixed(4));


	//$('#precision-value').text(history.val_precision[lastEpoch].toFixed(4));


	//$('#recall-value').text(history.val_recall[lastEpoch].toFixed(4));
}

// Compare selected models
/*function compareModels() {
	const selectedIds = $('#model-select').val();
	if (selectedIds.length < 2) return;
  
	// Fetch comparison data
	$.get(`/api/compare/?run1=${selectedIds[0]}&run2=${selectedIds[1]}`, function(data) {
		// Update metrics table
		const metrics = ['accuracy', 'loss', 'precision', 'recall'];
		let tableHtml = '';
    
		metrics.forEach(metric => {
			const m1 = data.metrics[metric].run1;
			const m2 = data.metrics[metric].run2;
			const delta = (m2 - m1).toFixed(4);
      
			tableHtml += `
				<tr>
					<td>${metric.charAt(0).toUpperCase() + metric.slice(1)}</td>
					<td>${m1.toFixed(4)}</td>
					<td>${m2.toFixed(4)}</td>
					<td class="${delta > 0 ? 'text-success' : 'text-danger'}">
						${delta > 0 ? '+' : ''}${delta}
					</td>
				</tr>
			`;
		});
    
		$('#metrics-table tbody').html(tableHtml);
    
		// Update comparison charts
		updateComparisonChart('compare-acc-chart', 
			['Accuracy', 'Validation Accuracy'], 
			[data.training_history.accuracy.run1, data.training_history.val_accuracy.run1],
			[data.training_history.accuracy.run2, data.training_history.val_accuracy.run2]
		);
    
		updateComparisonChart('compare-loss-chart', 
			['Loss', 'Validation Loss'], 
			[data.training_history.loss.run1, data.training_history.val_loss.run1],
			[data.training_history.loss.run2, data.training_history.val_loss.run2]
		);
	});
}*/

var scenes = [];

var comparisonChart1 = null;
var comparisonChart2 = null;

function updateComparisonChart(canvasId, labels, run1Data, run2Data, number, i) {
	const ctx = document.getElementById(canvasId).getContext('2d');

	if (number == 1) {
		if (scenes[i].comparisonChart1) {
			scenes[i].comparisonChart1.destroy()
		}
	}
	else if (number == 2) {
		if (scenes[i].comparisonChart2) {
			scenes[i].comparisonChart2.destroy()
		}
	}

	const comparisonChart = new Chart(ctx, {
		type: 'line',
		data: {
			labels: Array.from({length: run1Data[0].length}, (_, i) => i+1),
			datasets: [
				{ label: `${labels[0]} (Run 1)`, data: run1Data[0], borderColor: 'blue' },
				{ label: `${labels[1]} (Run 1)`, data: run1Data[1], borderColor: 'lightblue' },
				{ label: `${labels[0]} (Run 2)`, data: run2Data[0], borderColor: 'green' },
				{ label: `${labels[1]} (Run 2)`, data: run2Data[1], borderColor: 'lightgreen' }
			]
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			title: {
				display: true,
				text: canvasId.includes('acc') ? 'Accuracy Comparison' : 'Loss Comparison',
			}
		}
	});

	if (number == 1) {
		scenes[i].comparisonChart1 = comparisonChart;
	}
	else if (number == 2) {
		scenes[i].comparisonChart2 = comparisonChart;
	}
}

const defaultMaterial = new THREE.MeshStandardMaterial({ color: 0x404040, depthTest: true, depthWrite: true });
const activatedMaterial = new THREE.MeshStandardMaterial({ color: 0xFFFFFF });

const defaultLineMaterial = new THREE.LineBasicMaterial({ color: 0x282828, depthTest: true, depthWrite: true, linewidth: 1, vertexColors: true });

var orbitControls = null;// = new THREE.TrackballControls(camera, renderer.domElement); /*OrbitControls(camera, renderer.domElement);*/
/*orbitControls.enableDamping = true;
orbitControls.dampingFactor = 0.02;
orbitControls.maxPolarAngle = Math.PI;
orbitControls.maxDistance = 10000;
orbitControls.minDistance = 0.1;
orbitControls.target.set(0,0,0);*/

//camera.position.y = 60;
//orbitControls.update();

var geometries = [];
var meshes = [];
var materials = [];
var lines = [];
var saved = [];

function relu(x) {
	return Math.max(0,x);
}

function leaky_relu(x) {
	return Math.max(0.1 * x, x);
}

function sigmoid(x) {
	return 1/(1+ Math.pow(Math.E, -x));
}

function tanh(x) {
	return Math.tanh(x);
}

function linear(x) {
	return 1;
}

function softmax(x) {
	return Math.pow(Math.E, x)/(Math.pow(Math.E, x)*3);
}

const learning_rate = 0.1;
const epochs = 30;

var trained = false;

function derivative(x, func) {
	if (func === 'relu') {
		if (x < 0) {
			return 0;
		}
		else {
			return 1;
		}
	}
	else if (func === 'leaky_relu') {

	}
	else if (func === 'sigmoid') {
		
	}
	else if (func === 'tanh') {

	}
	else if (func === 'linear') {

	}
	else if (func === 'softmax') {
		
	}
}

// don't think too much about it. function was replaced with change()
function train() {
	const dataset = [
		//[0 = {
		//	input: [[0] = { weight_sum: 1 }, [1] = { weight_sum: 0.5 }],
		//	output: [[0] = 1, [1] = 0, [2] = 0]
		//}
	]
	dataset[0] = { input: [], output: [] };
	dataset[0].input[0] = { weight_sum: 1 };
	dataset[0].input[1] = { weight_sum: 0 };
	dataset[0].input[2] = { weight_sum: 0 };
	dataset[0].output[0] = 1;
	dataset[0].output[1] = 0;
	dataset[0].output[2] = 0;

	dataset[1] = { input: [], output: [] };
	dataset[1].input[0] = { weight_sum: 0 };
	dataset[1].input[1] = { weight_sum: 1 };
	//dataset[1].input[2] = { weight_sum: 0 };
	dataset[1].output[0] = 0;
	dataset[1].output[1] = 1;
	dataset[1].output[2] = 0;

	dataset[2] = { input: [], output: [] };
	dataset[2].input[0] = { weight_sum: 0 }
	dataset[2].input[1] = { weight_sum: 0 };
	dataset[2].input[2] = { weight_sum: 1 };
	dataset[2].output[0] = 0;
	dataset[2].output[1] = 0;
	dataset[2].output[2] = 1;


	const layers = window.networkVisualizer.layers;

	for (let epoch = 0; epoch < epochs; epoch++) {
	for (let v = 0; v < dataset.length; v++) {
		change(dataset[v].input, true);

		var deltas = [];
		deltas[saved.length-1] = [];

		for (let i = 0; i < saved[saved.length-1].length; i++) {
			const error = dataset[v].output[i] - saved[saved.length-1][i].weight_sum;
			deltas[saved.length-1][i] = error * saved[saved.length-1][i].weight_sum * (1 - saved[saved.length-1][i].weight_sum);
		}

		for (let i = layers.length-2; i >= 1; i--) {
			deltas[i] = [];

			for (let j = 0; j < layers[i].units; j++) {
				var delta = derivative(saved[i][j].weight_sum, layers[i].activation)
				
				/*saved[i][j].weight_sum * (1 - saved[i][j].weight_sum);*/

				var error = 0;

				for (let k = 0; k < layers[i+1].units; k++) {
					error += deltas[i+1][k] * saved[i+1][k].weight;
				}

				deltas[i][j] = delta * error;

				//saved[i][j].weight += learning_rate * delta * saved[i][j].weight_sum;
			}
		}

		for (let i = 1; i < layers.length; i++) {
			for (let j = 0; j < layers[i].units; j++) {
				//var bias += learningRate * deltas[i][j];

				for (let k = 0; k < layers[i-1].units; k++) {
					//var prev = deltas[i-1][k]; 

					deltas[i][j] = learning_rate * deltas[i][j] * saved[i-1][k].weight_sum;
					saved[i][j].weight += deltas[i][j];
					//console.log(deltas[i-1][k])
					//saved[i-1][k].weight += 0.5 * prev;
				}
			}
		}

		// [weight_sum]
		// [weight, weight_sum][]
		// [weight, weight_sum][][]

		for (let i = 0; i < layers.length; i++) {
			for (let j = 0; j < layers[i].units; j++) {
				saved[i][j].weight_sum = 0;
			}
		}
	}
}

	/*for (let i = 1; i < layers.length-1; i++) {
		deltas[i] = [];

		for (let j = 0; j < layers[i].units; j++) {
			var delta = saved[i+1].weight_sum * (1 - saved[i+1].weight_sum);
			
			for (let k = 0; k < layers[i+1].units; k++) {
				delta += deltas[i+1][j] * saved[i].weight_sum
			}
		}
	}*/

	trained = true;
}

function change(indexes, training, meshes, geometries, lines, saved, id2) {
	const layers = window.networkVisualizer.layers;
	//const layers = nodes[id2].visualizer.layers;

	for (let i = 0; i < indexes.length; i++) {
		saved[0][i].weight_sum = indexes[i].weight_sum;

		if (indexes[i].weight_sum > 0) {
			meshes[0][i].material = new THREE.MeshStandardMaterial({ color: 0xFFFFFF })
		}
		else {
			meshes[0][i].material.color = defaultLineMaterial.color;
		}
	}

	for (let i = 1; i < layers.length; i++) {
		for (let j = 0; j < layers[i].units; j++) {
			for (let k = 0; k < layers[i-1].units; k++) {
	
				lines[i][j].colors[k * 6 + 0] = 40/255;
				lines[i][j].colors[k * 6 + 1] = 40/255;
				lines[i][j].colors[k * 6 + 2] = 40/255;
				lines[i][j].colors[k * 6 + 3] = 40/255;
				lines[i][j].colors[k * 6 + 4] = 40/255;
				lines[i][j].colors[k * 6 + 5] = 40/255;
			}
		}
	}

	//saved[0][index_].weight_sum = 0.5;
	//saved[0][2].weight_sum = 1
	//meshes[0][index_].material = new THREE.MeshStandardMaterial({ color: 0xFFFFFF })
	
	//var max_ = 0;

	for (let i = 1; i < layers.length; i++) {
		for (let j = 0; j < layers[i].units; j++) {
			var max_ = 0;
			var index = 1;
			while (true) {
				if (layers[i-index].type === 'dropout' || layers[i-index].type === 'batchnorm') {
					index += 1
				}
				else {
					break;
				}
			}

			for (let k = 0; k < layers[i-index].units; k++) {
				if (saved[i-index][k].weight_sum > 0) {
					//if (materials) {

					//}

					var x = saved[i][j].weight * saved[i-index][k].weight_sum;

					//console.log(saved[i][j].weight, saved[i-index][k].weight_sum);

					//console.log(x, layers[i].type);

					var y = 0;

					if (layers[i].activation === 'relu') {
						y = relu(x);
					}
					else if (layers[i].activation === 'leaky_relu') {
						y = leaky_relu(x);
					}
					else if (layers[i].activation === 'sigmoid') {
						y = sigmoid(x);
					}
					else if (layers[i].activation === 'tanh') {
						y = tanh(x);
					}
					else if (layers[i].activation === 'linear') {
						y = linear(x);
					}
					else if (layers[i].activation === 'softmax') {
						y = softmax(x);
					}

					if (layers[i].type !== 'output') {
						y = relu(x);
					}
					else {
						y = sigmoid(x);
					}
					
					saved[i][j].weight_sum += y
					
					max_ = Math.max(max_, saved[i][j].weight_sum)

					//color = getRGB(layers[i-1][j].weight);

					//meshes[i][j].
				}
			}

			var weight = max_ > 0 ? saved[i][j].weight_sum / max_ : saved[i][j].weight_sum;
			
			var color = getRGB(weight);
			meshes[i][j].material = new THREE.MeshStandardMaterial({ color: color });

			color = getRGB2(weight);

			for (let k = 0; k < layers[i-1].units /*lines[i][j].length*/; k++) {
				//for (let k = 0; k < layers[i-index].units; k++) {
					if (saved[i-1][k].weight_sum > 0) {
						//lines[i][j][k].material = new THREE.LineBasicMaterial({ color: color })
						lines[i][j].colors[k * 6 + 0] = color.r / 255;
						lines[i][j].colors[k * 6 + 1] = color.g / 255;
						lines[i][j].colors[k * 6 + 2] = color.b / 255;

						lines[i][j].colors[k * 6 + 3] = color.r / 255;
						lines[i][j].colors[k * 6 + 4] = color.g / 255;
						lines[i][j].colors[k * 6 + 5] = color.b / 255;

						//lines[i][j].geometry.addAttribute(new THREE.BufferAttribute('color', lines[i][j].colors))
						lines[i][j].geometry.attributes.color.needsUpdate = true;
					}
				//}
			}

			/*if (i < layers.length-1) {
				if (layers[i+1].type === 'dropout') {
					for (let k = 0; k < lines[i+1][0].length; k++) {
						if (saved[i][k].weight_sum > 0) {
							lines[i+1][0][k].material = new THREE.LineBasicMaterial({ color: color });
						}
					}
				}
			}*/
			
			//meshes[i][j];

			//saved[i][j].weight = saved[i][j].weight_sum;// / layers[i-index].units;
			//saved[i][j].weight_sum = 0;

			//layers[i][j].weight_sum = 0;
		}
	}

	if (!training) {
		for (let i = 0; i < layers.length; i++) {
			for (let j = 0; j < layers[i].units; j++) {
				saved[i][j].weight_sum = 0;
				saved[i][j].weight = Math.round(Math.random()) == 1 ? Math.random() / 0.3 : 0;
			}
		}
	}

	if (window.networkVisualizer.layers[1]) {

	}
}

function getRGB(weight) {
	if (weight < 0.2) {
		return 0x282828;
	}
	else if (weight >= 0.2 && weight < 0.4) {
		return 0x606060;
	}
	else if (weight >= 0.4 && weight < 0.8) {
		return 0x75a95ff;
	}
	else if (weight >= 0.8) {
		return 0x2ecc71;
	}
}

function getRGB2(weight) {
	if (weight < 0.2) {
		return { r: 40, g: 40, b: 40}; //0x000000;
	}
	else if (weight >= 0.2 && weight < 0.4) {
		return { r: 60, g: 60, b: 60}; //0x404040;
	}
	else if (weight >= 0.4 && weight < 0.8) {
		return { r: 122, g:149, b: 255}; //0x606060;
	}
	else if (weight >= 0.8) {
		return { r:45, g:205, b:144}; //0xFFFFFF;
	}
}

function convertColor(r,g,b) {
	r = Math.max(0, r - (r / 100));
	g = Math.max(0, g - (g / 100));
	b = Math.max(0, b - (b / 100));

	return { r: r,g: g,b: b }
}

const interval = 40 * 60; //25 * 60;

var last_time = 0;
var timer = 0;

var data = []
data[0] = []
data[0][0] = { weight_sum: 1 };
data[0][1] = { weight_sum: 1 };
data[0][2] = { weight_sum: 0 }
data[0][3] = { weight_sum: 0 }
data[0][4] = { weight_sum: 0 }
data[0][5] = { weight_sum: 0 }
data[0][6] = { weight_sum: 0 }
data[0][7] = { weight_sum: 0 }

data[1] = []
data[1][0] = { weight_sum: 0 }
data[1][1] = { weight_sum: 1 }
data[1][2] = { weight_sum: 0 }
data[1][3] = { weight_sum: 0 }
data[1][4] = { weight_sum: 0 }
data[1][5] = { weight_sum: 0 }
data[1][6] = { weight_sum: 1 }
data[1][7] = { weight_sum: 1 }

data[2] = []
data[2][0] = { weight_sum: 0 }
data[2][1] = { weight_sum: 0 }
data[2][2] = { weight_sum: 0 }
data[2][3] = { weight_sum: 1 }
data[2][4] = { weight_sum: 1 }
data[2][5] = { weight_sum: 0 }
data[2][6] = { weight_sum: 1 }
data[2][7] = { weight_sum: 0 }

data[3] = [];
data[3][0] = { weight_sum: 0 };
data[3][1] = { weight_sum: 1 };
data[3][2] = { weight_sum: 1 }
data[3][3] = { weight_sum: 1 }
data[3][4] = { weight_sum: 1 }
data[3][5] = { weight_sum: 0 }
data[3][6] = { weight_sum: 0 }
data[3][7] = { weight_sum: 0 }

data[4] = [];
data[4][0] = { weight_sum: 0 };
data[4][1] = {  };
data[4][2] = {  };
data[4][3] = {  };
data[4][4] = {  };
data[4][5] = {  };
data[4][6] = {  }
data[4][7] = {  }

var modeling = false;

var isRotating = true;
var rotationAngle = 0;

//var scenes = [];

function render(time) {
	//orbitControls.update();
	//renderer.render(scene, camera);

	for (let i = 0; i < scenes.length; i++) {

	if (time) {

		var delta = time - scenes[i].last_time;

		scenes[i].timer += 1 * Math.floor(delta);
		if (scenes[i].timer > interval) {
			if (modeling) {
				scenes[i].timer = 0;
			}
			else {
			//console.log("time to change");
			//change(1)
			//if (!trained) {
			//	train();
			//}
			//else {
			var index = Math.floor(Math.random() * (data.length - 1));
				//if (nodes[scenes[i].nodeid]) {
				//var data = [];
				//data[0] = { weight_sum: 1 };
				//data[1] = { weight_sum: 0 };
				//data[2] = { weight_sum: 0 };
				change(data[index], false, scenes[i].meshes, scenes[i].geometries, scenes[i].lines, scenes[i].saved, scenes[i].nodeid);
				//}
			//}
			scenes[i].timer = 0;
}
		}

		if (scenes[i].isRotating) {
			//rotationAngle += 0.05;
			//pivot.rotation.y += 0.05;// rotationAngle;

			const distance = scenes[i].camera.position.distanceTo(new THREE.Vector3(0,0,0));
			const angle = Math.atan2(camera.position.z, camera.position.x) + 0.0015;

			scenes[i].camera.position.x = distance * Math.cos(angle);
			scenes[i].camera.position.z = distance * Math.sin(angle);
			scenes[i].camera.lookAt(pivot.position);
		}

		scenes[i].last_time = time;
	}

	if (scenes[i].orbitControls.enabled) {
		scenes[i].orbitControls.update();
		}
	scenes[i].renderer.render(scenes[i].scene,scenes[i].camera);
	}

	requestAnimationFrame(render);
}

var size = 0;

function init(canvas, id2) {
	var add = {};

	add.scene = new THREE.Scene();
	add.renderer = new THREE.WebGLRenderer({ antialias: true });

	add.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
	canvas.appendChild(add.renderer.domElement);

	//canvas.querySelector("div").style = `width: ${canvas.clientWidth}; height: ${clientHeight}; `;
	//var width = `${canvas.clientWidth}px`
	//var height = `${canvas.clientHeight}px`
	//canvas.querySelector("div").style.height = height//`${canvas.clientHeight}px;`;
	//canvas.querySelector("div").style.width = width//`${canvas.clientWidth}px;`;

	//var add = {};

	//add.scene = new THREE.Scene()
	//add.renderer = new THREE.WebGLRenderer({ antialias: true });

	add.camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 8000 /*0.000001, 1000*100*/);
	add.camera.position.set(1000,0,size/2);

	add.camera.name = "camera";

	add.orbitControls = new THREE.TrackballControls(add.camera, add.renderer.domElement);
	add.orbitControls.target.set(0,0,size/2);
	add.orbitControls.update();

	const ambient = new THREE.AmbientLight(0xFFFFFF);
	add.scene.add(ambient);

	const light = new THREE.PointLight(0xFFFFFF, 0.5, 1000);
	add.light = light;
	add.light.name = "light";

	add.scene.add(light);

	add.timer = 0;
	add.last_time = 0;

	add.isRotating = false;

	add.geometries = [];
	add.meshes = [];
	add.lines = [];
	add.saved = [];

	add.nodeid = id2;

	scenes.push(add);

	var id = 0;

	for (let i = 0; i < scenes.length; i++) {
		if (scenes[i].scene == add.scene) {
			id = i;
			break;
		}
	}

	//scene.add(pivot);
	
	//canvas.querySelector("canvas").addEventListener('mousedown', e => {
	//	console.log("uhm")
	//	orbitControls.enabled = true;
	//	isRotating = false;
	//});

	//scene.add(new THREE.AxesHelper(500))

	return { scene: add.scene, light: add.light, geometries: [], meshes: [], lines: [], saved: [], id: id, id2: id2 };
}


// Neural Network Visualization
class NetworkVisualizer {
	constructor(canvasId, constructor, id2) {
		this.canvas = document.getElementById(canvasId);
		this.constructor = constructor;
		this.layers = [
		{ type: 'input', units: 8, label: 'Input Features', dropout: 0.3, batch_norm_momentum: 0.99, weight: 0 },
		{ type: 'dense', units: 64, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.99, weight: 0 },
		{ type: 'dropout', units: 1, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.99, weight: 0 },
		{ type: 'dense', units: 32, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.99, weight: 0 },
		{ type: 'output', units: 3, activation: 'softmax', dropout: 0.3, batch_norm_momentum: 0.99, weight: 0 }
    	];
		this.data = init(this.canvas, id2);

		this.render();

		render();
	}

	render() {
		modeling = true;

		this.canvas.parentElement.querySelectorAll(".visualizer canvas").forEach(e => {
			if (e.id === 'network-canvas') {
				return
			}

			e.remove()
			//document.getElementById('network-canvas').remove()
		});

		const objectCount = 25;
		const objectsPerRow = 5;
		const spacing = 50;
		const centerZ = 0;

		const neuronSize = 8;

		const layerSpacing = 200;
		var prev = 0;

		for (let i = this.data.scene.children.length-1; i >= 0; i--) {
			const object = this.data.scene.children[i];

			if (object.isMesh) {
				if (object.geometry) {
					object.geometry.dispose();
				}
				if (object.material) {

				}
				this.data.scene.remove(object);
			}

			if (object.name === 'lines') {
				object.geometry.dispose();
				this.data.scene.remove(object);
			}
		}

		/*if (meshes.length > 0 && lines.length > 0) { 
		for (let i = 1; i < this.layers.length; i++) {
			for (let j = 0; j < this.layers[i].units; j++) {
				console.log(lines[i], i,j)
				if (meshes[i] && meshes[i][j]) {
				scene.remove(meshes[i][j]);
				}

				if (lines[i] && lines[i][j]) {
				scene.remove(lines[i][j].mesh);
				}

				//meshes[i][j].dispose();
				//geometries[i][j].dispose();
				//lines[i][j].geometry.dispose();
			}
		}
	}*/

		this.layers.forEach((layer, i) => {
			//model.add(new TSP.layers.Input);

			this.data.meshes[i] = [];
			this.data.geometries[i] = [];
			this.data.saved[i] = [];
			this.data.lines[i] = [];

			size += neuronSize + layerSpacing;

			/*var input_positions = [
				[0] = { x: 0, y: 0, z: 0 },
				[1] = { x: 0, y: 1, z: 0 },
				[2] = { x: 1, y: 0, z: 0 },
				[3] = { x: 1, y: 1, z: 0 },
				[4] = { x:  }
			] */

			for (let j = 0; j < layer.units; j++) {
				/*const geometry = new THREE.IcosahedronGeometry(6, 5);*/
				var val = 20;
				var spacing_ = layerSpacing;

				var objectsPerRow = Math.floor(Math.sqrt(layer.units));

				const row = Math.floor(j / objectsPerRow);
				const col = j % objectsPerRow;

				if (layer.units >= 128) {
					val = 30;
					spacing_ = 300
				}
				else if (layer.units >= 256) {
					val = 50
					spacing_ = 450
				}
				else if (layer.units >= 512) {
					val = 80
					spacing_ = 600
				}

				const x = (col - (objectsPerRow - 1) / 2) * spacing;
				const y = (row - (layer.units / objectsPerRow - 1) / 2) * spacing;
				const z = layerSpacing * i + centerZ;

				/*if (layer.units >= 128) {
					val = 10
				}
				else if (layer.units >= 256) {
					val = 30
				}
				else if (layer.units >= 512) {
					val = 80
				}*/

				const dx = layer.type !== 'input' && layer.type !== 'output' ? Math.random() * val - (neuronSize / 2) : 0;
				const dy = layer.type !== 'input' && layer.type !== 'output' ? Math.random() * val - (neuronSize / 2) : 0;
				const dz = layer.type !== 'input' && layer.type !== 'output' ? Math.random() * val - (neuronSize / 2) : 0;

				this.data.saved[i][j] = { x: x+dx, y: y+dy, z: z+dz, weight: Math.round(Math.random()) == 1 ? Math.random() / 0.3 : 0, weight_sum: 0 };
				
				if (layer.type === 'dropout') {
					this.data.saved[i][j].weight = 0.5;
				}

				var data = []
				data[0] = 0.8;
				data[1] = 0.5;
				data[2] = 0.2;

				if (layer.type === 'output') {
					this.data.saved[i][j].weight = Math.round((Math.random() * 10)) == 8 ? 0.8 : 0;
				}

				var dx2 = 0;
				var dy2 = 0;
				var dz2 = 0;

				if (layer.type === 'input') {
					if (j == 0) {
						dx2 = -6 * neuronSize;
						dy2 = -3 * neuronSize;
					}
					else if (j == 1) {
						dx2 = 6 * neuronSize;
						dy2 = -3 * neuronSize;
					}
					else if (j == 6) {
						dx2 = -6 * neuronSize;
						dy2 = 3 * neuronSize;
					}
					else if (j == 7) {
						dx2 = 6 * neuronSize;
						dy2 = 3 * neuronSize;
					}
				}

				this.data.lines[i][j] = [];
				this.data.meshes[i][j] = []
				
				if (i > 0) {
					/*if (this.layers[i-1].type === 'input') {
						if (j == 0) {
							dx2 = -12;
							dy2 = -6;
						}
						else if (j == 1) {
							dx2 = 12;
							dy2 = -6;
						}
						else if (j == 6) {
							dx2 = -12;
							dy2 = 6
						}
						else if (j == 7) {
							dx2 = -12;
							dy2 = 6;
						}
					}*/

					const positions = new Float32Array(this.data.saved[i-1].length * 2 * 3);
					const colors = new Float32Array(this.data.saved[i-1].length * 2 * 3);

					const geometry = new THREE.BufferGeometry();

					for (let k = 0; k < this.data.saved[i-1].length; k++) {
						const x2 = this.data.saved[i-1][k].x;
						const y2 = this.data.saved[i-1][k].y;
						const z2 = this.data.saved[i-1][k].z;

						const v1 = new THREE.Vector3(x+dx,y+dy,z+dz-(neuronSize / 2));
						const v2 = new THREE.Vector3(x2,y2,z2+(neuronSize / 2));

						const arr = [];
						arr.push(v1);
						arr.push(v2);

						if (this.layers[i-1].type === 'input') {
							if (k == 0) {
								dx2 = -6 * neuronSize;
								dy2 = -3 * neuronSize;
							}
							else if (k == 1) {
								dx2 = 6 * neuronSize;
								dy2 = -3 * neuronSize;
							}
							else if (k == 6) {
								dx2 = -6*neuronSize;
								dy2 = 3 * neuronSize
							}
							else if (k == 7) {
								dx2 = 6 * neuronSize;
								dy2 = 3 * neuronSize
							}
							else {
								dx2 = 0;//3 * neuronSize;
								dy2 = 0;//3 * neuronSize;
							}
						}

						positions[k * 6 + 0] = x+dx;
						positions[k * 6 + 1] = y+dy;
						positions[k * 6 + 2] = z+dz-(neuronSize / 2);

						positions[k * 6 + 3] = x2+dx2;
						positions[k * 6 + 4] = y2+dy2;
						positions[k * 6 + 5] = z2+(neuronSize / 2)+dz2;

						colors[k * 6 + 0] = 1;
						colors[k * 6 + 1] = 1;
						colors[k * 6 + 2] = 1;
						colors[k * 6 + 3] = 1;
						colors[k * 6 + 4] = 1;
						colors[k * 6 + 5] = 1;

						//const geometry2 = new THREE.BufferGeometry().setFromPoints(arr);
						//const line = new THREE.Line(geometry2, defaultLineMaterial);

						//scene.add(line);

						
						//lines[i][j][k] = colors;
					}

					geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
					geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));

					const lineSegments = new THREE.LineSegments(geometry, defaultLineMaterial);
					lineSegments.name = "lines";
					this.data.scene.add(lineSegments);

					this.data.lines[i][j] = { colors: colors, geometry: geometry, mesh: lineSegments };
				}

				if (layer.type === 'input') {
					const geometry = new THREE.BoxBufferGeometry(16,16,16);
					const box = new THREE.Mesh(geometry, defaultMaterial);
					box.name = j;

					this.data.geometries[i][j] = geometry;
					this.data.meshes[i][j] = box;

					/*var dx2 = 0;
					var dy2 = 0;
					var dz2 = 0;

					if (j == 0) {
						dx2 = -12;
						dy2 = -6;
					}
					else if (j == 1) {
						dx2 = 12;
						dy2 = -6;
					}
					//else if (j == 2) {
					//	dx2 = -6;
					//	dy2 = -6;
					//}  
					else if (j == 6) {
						dx2 = -12;
						dy2 = 6;
					}
					else if (j == 7) {
						dx2 = 12
						dy2 = 6
					}*/

					box.position.set(x+dx2,y+dy2,z+dz2);

					this.data.scene.add(box)
				}
				else if (layer.type === 'output') {
					const geometry = new THREE.BoxBufferGeometry(16,16,16);
					const box = new THREE.Mesh(geometry, defaultMaterial);
					box.name = j;

					this.data.geometries[i][j] = geometry;
					this.data.meshes[i][j] = box;

					box.position.set(x,y,z);

					this.data.scene.add(box)
				}
				else {

					const geometry = new THREE.BoxBufferGeometry(neuronSize, neuronSize, neuronSize);
					const sphere = new THREE.Mesh(geometry, activatedMaterial);
					sphere.name = j;


					this.data.geometries[i][j] = geometry;
					this.data.meshes[i][j] = sphere;

					sphere.position.set(x+dx,y+dy,z+dz)

					this.data.scene.add(sphere);
				}
			}

			for (let j = 0; j < layer.units; j++) {
				
			}

			this.data.light.position.set(0,0,size/2);
		});

		var id_ = this.data.id;
		scenes[id_].geometries = this.data.geometries;
		scenes[id_].meshes = this.data.meshes;
		scenes[id_].lines = this.data.lines;
		scenes[id_].saved = this.data.saved;
		scenes[id_].nodeid = this.data.id2;

		modeling = false;
	}
  
	getLayerColor(type) {
		const colors = {
			'input': '#3498db',
			'dense': '#2ecc71',
			'dropout': '#e74c3c',
			'batchnorm': '#9b59b6',
			'output': '#f1c40f'
		};
		return colors[type] || '#95a5a6';
	}
  
	updateLayer(index, units, activation,dropout,batch_norm_momentum,type) {
		if (this.layers[index]) {
			if (units) {
				this.layers[index].units = units;
			}
			if (dropout != null) {
				this.layers[index].dropout = dropout
			}
			if (batch_norm_momentum != null) {
				this.layers[index].batch_norm_momentum = batch_norm_momentum
			}
			if (type) {
				this.layers[index].type = type
			}
			if (activation) this.layers[index].activation = activation;
			this.render();
		}
	}
  
	addLayer(type = 'dense', units = 32, activation = 'relu', dropout=0.3, batch_norm_momentum=0.99, weight=0) {
    // Insert before output layer
    this.layers.splice(this.layers.length - 1, 0, {
      type,
      units,
      activation,
	  dropout,
	  batch_norm_momentum, weight
    });
    this.render();
    return this.layers.length - 2; // Return index of new layer
  }
  
  removeLayer(index) {
    if (index > 0 && index < this.layers.length - 1) {
      this.layers.splice(index, 1);
      this.render();
      return true;
    }
    return false;
  }
  
  getArchitecture() {
    return this.layers.map(layer => ({
      type: layer.type,
      units: layer.units,
      activation: layer.activation,
	  dropout: layer.dropout,
	  batch_norm_momentum: layer.batch_norm_momentum
    }));
  }
}


//####

// Layer controls manager
class LayerControls {
	constructor(containerId, visualizer, constructor) {
		this.container = document.getElementById(containerId);
		this.visualizer = visualizer;
		this.constructor = constructor;
		this.render();
	}
  
	render() {
		this.container.innerHTML = '';
    
		// Skip input and output layers
		const hiddenLayers = this.visualizer.layers.slice(1, -1);
    
		hiddenLayers.forEach((layer, index) => {
			const layerIndex = index + 1; // Actual index in visualizer
			const layerEl = document.createElement('div');
			const index_ = index + 1 < 10 ? "0" + (index+1).toString() : index+1;
			layerEl.className = 'layer-control mb-3 p-3 border rounded purple-border-primary';
			layerEl.innerHTML = `
				<div class="accent">
				<span class="accent-span"></span>
				<span class="accent-span-2"></span>
				<span class="accent-span-3"></span>
				<div class="layer-control-inner">
				<div class="d-flex justify-content-between align-items-center mb-2">
					<h4 class="mb-0">LAYER // ${index_}</h5>
					<button class="btn btn-sm btn-danger" onclick="removeLayer(${layerIndex})">
						<i class="fas fa-trash"></i>
					</button>
				</div>
				<div class="form-group">
					<label>Units</label>
					<input type="range" class="form-control-range layer-units" min="8" max="512" step="1" value="${layer.units}" data-layer="${layerIndex}">
					<output class="ml-2">${layer.units}</output>
				</div>
				<div class="form-group">
					<label>Activation</label>
					<select class="form-control layer-activation" data-layer="${layerIndex}">
						<option value="relu" ${layer.activation === 'relu' ? 'selected' : ''}>ReLU</option>
						<option value="leaky_relu" ${layer.activation === 'leaky_relu' ? 'selected' : ''}>Leaky ReLU</option>
						<option value="sigmoid" ${layer.activation === 'sigmoid' ? 'selected' : ''}>Sigmoid</option>
						<option value="tanh" ${layer.activation === 'tanh' ? 'selected' : ''}>Tanh</option>
						<option value="linear" ${layer.activation === 'linear' ? 'selected' : ''}>Linear</option>
					</select>
				</div>
				<div class="form-group">
					<label>Regularization</label>
					<div class="d-flex">
						<div class="custom-control custom-checkbox mr-3">
							<input type="checkbox" class="custom-control-input layer-dropout" id="dropout-${layerIndex}" ${layer.type === 'dropout' ? 'checked' : ''} data-layer="${layerIndex}">
							<label class="custom-control-label" for="dropout-${layerIndex}">Dropout</label>
						</div>
						<div class="custom-control custom-checkbox">
							<input type="checkbox" class="custom-control-input layer-batch-norm" id="batchnorm-${layerIndex}" ${layer.type === 'batchnorm' ? 'checked' : ''} data-layer="${layerIndex}">
							<label class="custom-control-label" for="batchnorm-${layerIndex}">BatchNorm</label>
						</div>
					</div>
					<div class="d-flex dropout mt-3" style="${layer.type === 'dropout' ? '' : 'display: none;'}">
						<div class="custom-control d-flex">
							<label class="custom-control-label">Dropout Rate</label>
							<input type="number" class="custom-control-input dropout-rate" step="0.01" min="0" max="0.9" value="${layer.dropout}" data-layer="${layerIndex}">
						</div>
					</div>
					<div class="d-flex batchnorm mt-3" style="${layer.type === 'batchnorm' ? '' : 'display: none;'}">
						<div class="custrom-control d-flex">
							<label class="custom-control-label">Momentum</label>
							<input type="number" step="0.01" min="0" max="0.99" value="${layer.batch_norm_momentum}" data-layer="${layerIndex}">
						</div>
					</div>
				</div>
				</div>
				</div>
			`;
			this.container.appendChild(layerEl);
		});

		var button = document.createElement("button");
		button.className = "layer-control btn-large";
		button.innerHTML = `
			+
		`;
		button.addEventListener('click', e => {
			addLayer(e);
		});

		this.container.appendChild(button);
    
		// Add event listeners
		document.querySelectorAll('.layer-units').forEach(slider => {
			slider.addEventListener('input', e => {
				const layerIndex = parseInt(e.target.dataset.layer);
				const units = parseInt(e.target.value);
				e.target.nextElementSibling.textContent = units;
				this.visualizer.updateLayer(layerIndex, units);
			});
		});
    
		document.querySelectorAll('.layer-activation').forEach(select => {
			select.addEventListener('change', e => {
				const layerIndex = parseInt(e.target.dataset.layer);
				this.visualizer.updateLayer(layerIndex, null, e.target.value);
			});
		});

		document.querySelectorAll('.layer-dropout').forEach(select => {
			select.addEventListener('change', e => {
				const layerIndex = parseInt(e.target.dataset.layer);

				var checked = e.target.parentElement.parentElement.querySelector(".custom-checkbox .layer-batch-norm").checked

				if (!checked) {
					var type = 'dense';

					var dropout = e.target.parentElement.parentElement.parentElement.querySelector(".dropout");

					if (e.target.checked) {
						type = 'dropout';
						
						//this.visualizer.updateLayer(layerIndex,null,null,e.target.value,null,'dropout');
						dropout.style.display = "block";
					}
					else {
						dropout.style.display = "none";
					}

					//e.target.parentElement.parentElement.parentElement.querySelector(".dropout").style.display = "block";

					//else {
					this.visualizer.updateLayer(layerIndex,null,null,parseFloat(dropout.querySelector(".custom-control .dropout-rate").value),null,type);
				}			
				else {
					e.target.checked = false;
				}
				
				//}
			});
		});

		document.querySelectorAll('.dropout-rate').forEach(select => {
			select.addEventListener('change', e => {
				const layerIndex = parseInt(e.target.dataset.layer);

				this.visualizer.updateLayer(layerIndex,null,null,parseFloat(e.target.value),null,null);
			});
		});

		document.querySelectorAll('.layer-batch-norm').forEach(select => {
			select.addEventListener('change', e => {
				const layerIndex = parseInt(e.target.dataset.layer);

				var checked = e.target.parentElement.parentElement.querySelector(".custom-checkbox .layer-dropout").checked;

				if (!checked) {
					var type = 'dense';

					var batchnorm = e.target.parentElement.parentElement.parentElement.querySelector(".batchnorm");

					if (e.target.checked) {
						type = 'batchnorm';

						batchnorm.style.display = "block";
					}
					else {
						batchnorm.style.display = "none";
					}

					this.visualizer.updateLayer(layerIndex,null,null,parseFloat(e.target.checked),null,type);
				}
				else {
					e.target.checked = false
				}

				//this.visualizer.updateLayer(layerIndex, null, null, null, e.target.value)
				//console.log("f");
			});
		});

		document.querySelectorAll('.batch-norm-momentum').forEach(select => {
			select.addEventListener('change', e => {
				const layerIndex = parseInt(e.target.dataset.layer);

				this.visualizer.updateLayer(layerIndex,null,null,null, parseFloat(e.target.value),null);
			});
		});
	}
}

// Visualization enhancements
function initNetworkVisualization(constructor,i) {
	// Initialize visualizer
	if (!constructor) {
	const visualizer = new NetworkVisualizer('canvas');
	const layerControls = new LayerControls('layers', visualizer);

	window.networkVisualizer = visualizer
	window.layerControls = layerControls
	constructor = document;
	}

	// Setup mode switching
	/*document.getElementById('simple-mode-btn').addEventListener('click', () => {
		document.getElementById('simple-mode').style.display = 'block';
		document.getElementById('advanced-mode').style.display = 'none';
		//this.classList.add('active');
		document.getElementById('simple-mode-btn').classList.add('active')
		document.getElementById('advanced-mode-btn').classList.remove('active');
	});*/
  
	/*document.getElementById('advanced-mode-btn').addEventListener('click', () => {
		document.getElementById('simple-mode').style.display = 'none';
		document.getElementById('advanced-mode').style.display = 'block';
		//this.classList.add('active');
		document.getElementById('advanced-mode-btn').classList.add('active')
		document.getElementById('simple-mode-btn').classList.remove('active');
    
		// Initialize 3D-like perspective
		//apply3DPerspective();
	});*/

	console.log(constructor);

	constructor.querySelector('#optimizer').addEventListener('change', e => {
		var adam = constructor.querySelector('#adam-params')
		var sgd = constructor.querySelector('#sgd-params')
 
		if (e.target.value === 'adam') {
			adam.style.display = "block";
			sgd.style.display = "none";
		}
		else if (e.target.value === 'rmsprop') {
			adam.style.display = "none";
			sgd.style.display = "none";
		}
		else if (e.target.value === 'sgd') {
			adam.style.display = "none";
			sgd.style.display = "block";
		}
	});

	constructor.querySelector('.container-header').addEventListener('click', e => {
		const container = constructor.querySelector('container-window');

		if (container.style.display === "") {
			container.style.display = "none";
		}
		else {
			container.style.display = "block";

			if (!scenes[i].accChart && !scenes[i].lossChart) {
				scenes[i].accChart = new Chart($('#accuracy-chart'), {
					type: 'line',
					data: { datasets: [
						{ label: 'Training Accuracy', data: [   ], borderColor: 'blue' },
						{ label: 'Validation Accuracy', data: [   ], borderColor: 'green' }
					] },
					options: { title: { display: true, text: 'Training Accuracy' }, responsive: true, maintainAspectRatio: false }
				});

				scenes[i].lossChart = new Chart($('#loss-chart'), {
					type: 'line',
					data: { datasets: [
						{ label: 'Training Loss', data: [   ], borderColor: 'red' },
						{ label: 'Validation Loss', data: [   ], borderColor: 'orange' }
					] },
					options: { title: { display: true, text: 'Training Loss' }, responsive: true, maintainAspectRatio: false }
				})
			}

			if (scenes[i].savedCharts) {
				updateTrainingCharts(scenes[i].savedCharts);
			}


			//container.style.display = "block";
		}
	});

	constructor.querySelector('.container-window #close-container-window').addEventListener('click', e => {
		const container = constructor.querySelector(".container-window");

		container.style.display = "none";
	});

	//constructor.querySelector('.sidebar-training #classification-results-spec button').addEventListener('click', e => {
	//	const container = constructor.querySelector("#container-window-results");

	//	container.style.display = "block";
	//})

	//constructor.querySelector('.container-window #close-container-window-results').addEventListener('click', e => {
	//	const container = constructor.querySelector("#container-window-results");

	//	container.style.display = "none";
	//});

	constructor.querySelectorAll('.nav-link').forEach(select => {
		select.addEventListener('click', e => {
			if (e.target.dataset.toggle === 'tab') {
				var current = constructor.querySelector('current-run');
				var comparison = constructor.querySelector('comparison');
				var history = constructor.querySelector('history');

				constructor.querySelectorAll('#trainingTabs .nav-item .nav-link').forEach(e2 => {
					e2.classList.remove('active')
				});

				e.target.classList.add('active')
				if (e.target.href.match('#current-run')) {
					current.style.display = "block";
					comparison.style.display = "none";
					history.style.display = "none";
				}
				else if (e.target.href.match('#comparison')) {
					current.style.display = "none";
					comparison.style.display = "block";
					history.style.display = "none";

					if (!scenes[i].comparisonChart1 && !scenes[i].comparisonChart2) {
						updateComparisonChart('compare-acc-chart', ["Accuracy", "Validation Accuracy"], [[],[]], [[],[]], 1, i)
						updateComparisonChart('compare-loss-chart', ["Loss", "Validatoin Loss"], [[],[]], [[],[]], 2 ,i)
					}
				}
				else if (e.target.href.match('#history')) {
					current.style.display = "none";
					comparison.style.display = "none";
					history.style.display = "block";
				}
			}
		});
	});
	
  
}

var networknodes = []

// Layer management functions
function addLayer(id=0, type='dense', units=64, activation='relu', dropout=0.3, batchnorm_momentum=0.99) {
	//var id = 0;

	if (document.URL.match('sandbox')) {

	const visualizer = window.networkVisualizer;

	const layers = document.getElementById('layers');

	//if (visualizer.layers.length == 22) {
	//	layers.scrollX = layers.clientWidth;
	//	return;
	//}
	
	const newIndex = visualizer.addLayer(type,units,activation,dropout,batchnorm_momentum);
  
	// Update layer controls
	//if (nodes[id].layer) {
	//	nodes[id].layer.render();
	//}
	window.layerControls.render()
  
	// Highlight new layer
	highlightLayer(newIndex);

	}
	else {
		//const visualizer = nodes[id].visualizer;
		//const layers = document.getElementById('layers');

		//if (visualizer.layers.length == 12) {
		//	return;
		//}

		const newIndex = visualizer.addLayer(type,units,activation,dropout,batchnorm_momentum,model);
		
		//const layerControls = nodes[id].layer

		//if (layerControls) {
		//	layerControls.render();
		//}
		window.layerControls.render()
	}
}

function removeLayer(index) {
	const visualizer = window.networkVisualizer;
	if (visualizer.removeLayer(index)) {// && !skip) {
		window.layerControls.render();
	}
}

function highlightLayer(index) {
	/*const visualizer = window.networkVisualizer
	
	const canvas = document.getElementById('network-canvas');
	const ctx = canvas.getContext('2d');
  
	// Get layer position
	const layer = visualizer.layers[index];
	const layerSpacing = canvas.width / (visualizer.layers.length + 1);
	const x = layerSpacing * (index + 1);
  
	// Draw highlight
	ctx.shadowColor = 'rgba(255,215,0,0.5)';
	ctx.shadowBlur = 20;
	ctx.fillStyle = 'rgba(255,255,255,0.1)';
	ctx.fillRect(x - 40, 0, 80, canvas.height);
	ctx.shadowBlur = 0;
  
	// Remove highlight after delay
 	setTimeout(() => visualizer.render(), 1000);*/
}

// Initialize on page load
//document.addEventListener('DOMContentLoaded', () => {
//	initNetworkVisualization();
//
//	document.getElementById('add-layer').addEventListener('click', addLayer())
//});

// Dimension visualization
class DimensionVisualizer {
	constructor(canvasId) {
		this.canvas = document.getElementById(canvasId);
		this.ctx = this.canvas.getContext('2d');
		this.dimensions = { x: 8, y: 64, z: 32 }; // Input, hidden, output
		this.rotation = { x: -20, y: 30 };
		this.render();
	}

	render() {
		const ctx = this.ctx;
		const width = this.canvas.width;
		const height = this.canvas.height;
		ctx.clearRect(0, 0, width, height);
    
		// Center of canvas
		const centerX = width / 2;
		const centerY = height / 2;
    
		// Convert rotation to radians
		const radX = this.rotation.x * Math.PI / 180;
		const radY = this.rotation.y * Math.PI / 180;
    
		// Cube vertices (normalized)
		const vertices = [
			{ x: -1, y: -1, z: -1 },
			{ x: 1, y: -1, z: -1 },
			{ x: 1, y: 1, z: -1 },
			{ x: -1, y: 1, z: -1 },
			{ x: -1, y: -1, z: 1 },
			{ x: 1, y: -1, z: 1 },
			{ x: 1, y: 1, z: 1 },
			{ x: -1, y: 1, z: 1 }
		];
    
		// Scale vertices based on dimensions
		const scale = 0.8 * Math.min(width, height) / 4;
		const scaledVertices = vertices.map(v => ({
			x: v.x * this.dimensions.x * scale / 8,
			y: v.y * this.dimensions.y * scale / 64,
			z: v.z * this.dimensions.z * scale / 32
		}));
    
		// Project 3D to 2D
		const projected = scaledVertices.map(v => {
			// Rotate around X axis
			const y1 = v.y * Math.cos(radX) - v.z * Math.sin(radX);
			const z1 = v.y * Math.sin(radX) + v.z * Math.cos(radX);
      
			// Rotate around Y axis
			const x2 = v.x * Math.cos(radY) + z1 * Math.sin(radY);
			const z2 = -v.x * Math.sin(radY) + z1 * Math.cos(radY);
      
			// Project to 2D
			return {
				x: centerX + x2,
				y: centerY - y1,
				depth: z2
			};
		});
    
    // Draw cube faces
    const faces = [
      { points: [0, 1, 2, 3], color: 'rgba(52, 152, 219, 0.7)' }, // Front
      { points: [4, 5, 6, 7], color: 'rgba(46, 204, 113, 0.7)' }, // Back
      { points: [0, 4, 7, 3], color: 'rgba(155, 89, 182, 0.7)' }, // Left
      { points: [1, 5, 6, 2], color: 'rgba(241, 196, 15, 0.7)' }, // Right
      { points: [0, 1, 5, 4], color: 'rgba(231, 76, 60, 0.7)' }, // Bottom
      { points: [3, 2, 6, 7], color: 'rgba(26, 188, 156, 0.7)' }  // Top
    ];
    
    // Draw faces in depth order
		faces.sort((a, b) => {
			const depthA = a.points.reduce((sum, i) => sum + projected[i].depth, 0) / a.points.length;
			const depthB = b.points.reduce((sum, i) => sum + projected[i].depth, 0) / b.points.length;
			return depthB - depthA; // Back to front
		});
    
		faces.forEach(face => {
			ctx.fillStyle = face.color;
			ctx.beginPath();
			face.points.forEach((vertexIdx, i) => {
				const v = projected[vertexIdx];
				if (i === 0) ctx.moveTo(v.x, v.y);
				else ctx.lineTo(v.x, v.y);
			});
			ctx.closePath();
			ctx.fill();
      
			// Add wireframe
			ctx.strokeStyle = '#2c3e50';
			ctx.lineWidth = 1;
			ctx.stroke();
		});
    
		// Draw dimension labels
		ctx.fillStyle = '#34495e';
		ctx.font = 'bold 14px Arial';
		ctx.textAlign = 'center';
    
		// X dimension (width)
		const midX1 = midpoint(projected[0], projected[1]);
		const midX2 = midpoint(projected[4], projected[5]);
		ctx.fillText(`Input: ${this.dimensions.x}`, (midX1.x + midX2.x)/2, (midX1.y + midX2.y)/2);
    
		// Y dimension (height)
		const midY1 = midpoint(projected[0], projected[3]);
		const midY2 = midpoint(projected[4], projected[7]);
		ctx.fillText(`Hidden: ${this.dimensions.y}`, (midY1.x + midY2.x)/2, (midY1.y + midY2.y)/2);
    
		// Z dimension (depth)
		const midZ1 = midpoint(projected[0], projected[4]);
		const midZ2 = midpoint(projected[1], projected[5]);
		ctx.fillText(`Output: ${this.dimensions.z}`, (midZ1.x + midZ2.x)/2, (midZ1.y + midZ2.y)/2);
	}
  
	rotate(x, y) {
		this.rotation.x = (this.rotation.x + x) % 360;
		this.rotation.y = (this.rotation.y + y) % 360;
		this.render();
	}
}

// Helper function
function midpoint(p1, p2) {
	return { x: (p1.x + p2.x)/2, y: (p1.y + p2.y)/2 };
}

/*function startAdvancedTraining() {


	// Get architecture configuration
	const architecture = window.networkVisualizer.getArchitecture();
  
	// Get hyperparameters
	const hyperparams = {
		learning_rate: parseFloat(document.querySelector('[name="learning_rate"]').value),
		epochs: parseInt(document.querySelector('[name="epochs"]').value),
		// ... other params
	};
  
	// Submit to server
	fetch('/api/train/advanced', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({
			mode: 'advanced',
			architecture,
			hyperparams
		})
	}).then(response => response.json()).then(data => {
		if (data.task_id) {
			monitorTrainingProgress(data.task_id);
    	}
  	});
}*/

function startSimpleTraining() {
    // Disable button during training
    const btn = document.querySelector('#simple-mode button');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Training...';
    
    // Show progress container
    document.getElementById('training-progress').style.display = 'block';
    
    // Get parameters
    const complexity = document.querySelector('[name="model_complexity"]').value;
    const learningRate = parseFloat(document.querySelector('[name="learning_rate"]').value);
    const intensity = parseInt(document.querySelector('[name="training_intensity"]').value);
    
    // Map complexity to units
    const complexityMap = {
        'small': 32,
        'medium': 64,
        'large': 128,
        'xlarge': 256
    };
    const units = complexityMap[complexity] || 64;
    
    // Calculate epochs based on intensity (50% intensity = 15 epochs, 100% = 30 epochs)
    const epochs = Math.max(5, Math.round(30 * (intensity / 100)));
    
	const form = new FormData();
	form.append('data', JSON.stringify({
		mode: 'simple',
		units: units,
		model_complexity: complexity,
		learning_rate: learningRate,
		epochs: epochs
	}));

    // Submit to server
    fetch('/api/train/', {
        method: 'POST',
        headers: {
			//'Content-Type': 'application/json',
			'x-csrftoken': document.querySelector('[name=csrfmiddlewaretoken]').value
		},
        body: form /*JSON.stringify({
			mode: 'simple',
            units: units,
			model_complexity: complexity,
            learning_rate: learningRate,
            epochs: epochs
        })*/
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            monitorTrainingProgress(data.task_id);
        } else {
            alert('Training failed to start: ' + (data.error || 'Unknown error'));
            resetTrainingUI();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to start training: ' + error.message);
        resetTrainingUI();
    });
}

function startAdvancedTraining() {
    // Disable button during training
    const btn = document.querySelector('.sidebar-training #train');
    btn.disabled = true;
    btn.innerHTML = `Training...` //'<span class="spinner-border spinner-border-sm"></span> Training...';
    
    // Show progress container
    //document.getElementById('training-progress').style.display = 'block';
    
    // Get architecture configuration
    const architecture = window.networkVisualizer.getArchitecture();
    
    // Get hyperparameters
    const hyperparams = {
        learning_rate: parseFloat(document.querySelector('[name="learning_rate"]').value),
        epochs: parseInt(document.querySelector('[name="epochs"]').value),
        batch_size: parseInt(document.querySelector('[name="batch_size"]').value),
        optimizer: document.querySelector('[name="optimizer"]').value,
		
		beta_1: document.querySelector('[name="beta_1"]').value,
		beta_2: document.querySelector('[name="beta_2"]').value,

		momentum: document.querySelector('[name="momentum"]').value,
		nesterov: document.querySelector('[name="use_nesterov"]').checked,

		use_reg: document.querySelector('[name="use_reg"]').checked,
		l2_reg: document.querySelector('[name="l2_reg"]').value,
		use_batchnorm: document.querySelector('[name="use_batchnorm"]').checked
    };
    
	console.log(document.querySelector('[name=csrfmiddlewaretoken]').value)

	const form = new FormData()
	form.append('data', JSON.stringify({
		mode: 'advanced',
		learning_rate: hyperparams.learning_rate,
		batch_size: hyperparams.batch_size,
		epochs: hyperparams.epochs,
		optimizer: hyperparams.optimizer,
		beta_1: hyperparams.beta_1,
		beta_2: hyperparams.beta_2,
		momentum: hyperparams.momentum,
		nesterov: hyperparams.nesterov,
		use_reg: hyperparams.use_reg,
		l2_reg: hyperparams.l2_reg,
		use_batchnorm: hyperparams.use_batchnorm,
		architecture: architecture,
	}));
	form.append('training_data', document.getElementById('training_data').files[0])

    // Submit to server
    fetch('/api/train/', {
        method: 'POST',
        headers: {
			//'Content-Type': 'application/json',
			"x-csrftoken": document.querySelector('[name=csrfmiddlewaretoken]').value
		},
        body: form /*JSON.stringify({
			mode: 'advanced',
			learning_rate: hyperparams.learning_rate,
			batch_size: hyperparams.batch_size,
			epochs: hyperparams.epochs,
			optimizer: hyperparams.optimizer,
            architecture: architecture,
			training_data: document.getElementById('training_data').files[0]
            //hyperparams: hyperparams
        })*/
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            monitorTrainingProgress(data.task_id);
        } else {
            alert('Training failed to start: ' + (data.error || 'Unknown error'));
            resetTrainingUI();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to start training: ' + error.message);
        resetTrainingUI();
    });
}

function startPredicting() {
	const btn = document.querySelector("#predict button");
	btn.disabled = true
	btn.innerHTML = 'Classifying...'

	const form = new FormData()
	form.append('datafile', document.getElementById('classification').files[0])

	if (!document.getElementById('classification').files[0]) {
		alert('No observation data.')

		btn.disabled = false;
		btn.innerHTML = "Classify Exoplanets";

		return
	}

	const id = document.querySelector('[name=current-model-id]').value;

	if (id === '') {
		alert('Please select trained model from training history or train a new model.')

		btn.disabled = false;
		btn.innerHTML = "Classify Exoplanets";
		return
	}

	fetch(`/api/predict/${id}/`, {
		method: 'POST',
		headers: {
			'x-csrftoken': document.querySelector('[name=csrfmiddlewaretoken]').value
		},
		body: form
	})
	.then(response => response.json())
	.then(data => {
		if (data.task_id) {
			monitorPredictingProgress(data.task_id)
		}
		else {
			alert('Predicting failed to start: ' + (data.error || 'Unknown error'));
			btn.disabled = false;
			btn.innerHTML = "Classify Exoplanets";
		}

	})
	.catch(error => {
		console.error('Error:', error);
		alert('Failed to start predicting: ' + error.message);
		btn.disabled = false;
		btn.innerHTML = "Classify Exoplanets";
	})
}

function monitorTrainingProgress(taskId) {
	const progressInterval = setInterval(() => {
		fetch(`/api/train/status/${taskId}`)
		.then(response => response.json())
		.then(data => {
			if (data.status == 'SUCCESS') {
				clearInterval(progressInterval);
				showTrainingResults(data.result);
				updateTrainingCharts(data.result.history);
				resetTrainingUI();

				document.getElementById('current-model').innerHTML = data.result.version;
				document.querySelector('[name=current-model-id]').value = data.result.current;

				alert('Model has been trained succesfully.');
			}
			else if (data.status == 'FAILURE') {
				clearInterval(progressInterval);
				showTrainingError(data.result);
				resetTrainingUI();
			}
			else if (data.status == 'PROGRESS') {
				updateTrainingProgressUI(data.progress);

				if (data.history) {
					updateTrainingCharts(data.history);
				}
			}
		})
		.catch(error => {
			console.error('Error checking status:', error);
			clearInterval(progressInterval);
			alert('Connection to training monitor failed.');
			resetTrainingUI();
		});
	}, 3000);
}

function updateTrainingProgressUI(progress) {
	//const progressBar = document.querySelector('#training-progress .progress-bar');
	//const percent = Math.round((progress.current_epoch / progress.total_epochs) * 100);
	//progressBar.style.width = `${percent}%`;
	//progressBar.textContent = `${percent}%`;

	//document.getElementById('current-epoch').textContent = progress.current_epoch;
	//document.getElementById('total-epochs').textContent = progress.total_epochs;

	document.getElementById('train').innerHTML = `Epoch: ${progress.current_epoch}/${progress.total_epochs}`;
}

function showTrainingResults(results) {
	//document.getElementById('training-progress').style.display = 'none';

	//$('#acc-value').text(results.metrics.accuracy.toFixed(4));
	//$('#loss-value').text(results.metrics.loss.toFixed(4));

	document.getElementById('train').innerHTML = 'Start Training';

	document.querySelectorAll('#accuracy-value').forEach(e => {
		e.innerHTML = results.metrics.accuracy.toFixed(4);
	});

	document.querySelectorAll('#loss-value').forEach(e => {
		e.innerHTML = results.metrics.loss.toFixed(4);
	})

	//document.querySelectorAll('#precision-value').forEach(e => {
	//	e.innerHTML = results.metrics.precision
	//});

	/*const resultsDiv = document.getElementById('training-results');
	resultsDiv.innerHTML = `
		<div class="alert alert-success mt-3">
			<h4>Training Complete!</h4>
			<p>Final Accuracy: ${(results.metrics.accuracy * 100).toFixed(2)}%</p>
			<p>Validation Loss: ${results.metrics.loss.toFixed(4)}</p>
			<p>Training Time: ${results.training_time.toFixed(4)} seconds</p>
			<p>Model Size: ${(results.model_size / 1024).toFixed(2)} KB</p>
			<p>You can now scroll below to use this model to predict Exoplanets</p>
		</div>
	`;*/

	//updateTrainingCharts()
}

function monitorPredictingProgress(taskId) {
	const btn = document.querySelector('#predict button');

	const progressInterval = setInterval(() => {
		fetch(`/api/predict/status/${taskId}`)
		.then(response => response.json())
		.then(data => {
			if (data.status == 'SUCCESS') {
				clearInterval(progressInterval)
				document.getElementById('classification-results').innerHTML = data.result.result.replace('\\n', '').replace('\\', '')
				btn.disabled = false;
				btn.innerHTML = 'Classify Exoplanets';
				document.getElementById('classification-check').innerHTML = 'Classification Complete'
			}
			else if (data.status == 'ERROR') {
				clearInterval(progressInterval)
				/*document.getElementById('classification-results').innerHTML = `
					<div class="alert alert-danger mt-3">
						<h4>Predicting Failed<h4>

						<p>${error.message || 'Unknown error occured during predicting'}</p>
						${error.message_issue ? `<p class="mtf-2"><strong>Resource Issue:</strong> ${error.resource_issue}</p>` : ''}
					</div>
				`;*/

				btn.disabled = false;
				btn.innerHTML = 'Classify Exoplanets';
			}
		})
		.catch(error => {
			console.error('Error checking status:', error)
			clearInterval(progressInterval)
			alert('Connection to prediction monitor failed.')
		});
	}, 3000);
}

function showTrainingError(error) {
	const resultsDiv = document.getElementById('training-results');
	resultsDiv.innerHTML = `
		<div class="alert alert-danger mt-3">
			<h4>Training Failed</h4>
			<p>${error.message || 'Unknown error occured during training'}</p>
			${error.message_issue ? `<p class="mt-2"><strong>Resource Issue:</strong> ${error.resource_issue}</p>` : ''}
		</div>
	`;
}

function resetTrainingUI() {
	document.querySelectorAll('#simple-mode .card .card-body button, #advanced-mode .container .col-md-5 .btn-primary').forEach(btn => {
		btn.disabled = false
		btn.innerHTML = btn === document.querySelector('#simple-mode button') ? 'Start Training' : 'Start Advanced Training'
	});
}

function resetPredictingUI () {
	const btn = document.querySelector('.predict button');
	btn.disabled = false
	btn.innerHTML = "Classify Exoplanets";
}

function loadModel(event, runId) {
	var id = 0;

	if (!document.URL.match('sandbox')) {
		id = parseInt(event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id.split('neural-network')[1]);
	}

    // Show loading indicator
    const btn = constructor.querySelector(`button[onclick="loadModel('${runId}')"]`);
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Loading';
    btn.disabled = true;

	//var id = 0;

	//if (!document.URL.match('sandbox')) {
	//	id = event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id.split('neural-network')[1];
	//}

	//var form = new FormData();
	//form.append('data', JSON.stringify({
	//	id: 
	//}));
    
    fetch(`/api/models/${runId}/load/`, {
        method: 'POST',
		headers: {
			'Content-Type': 'json/application',
			'x-csrftoken': document.querySelector('[name=csrfmiddlewaretoken]').value
		},
		//body: 
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update UI to show active model
			var constructor = nodes[id].constructor;

            document.querySelectorAll('.model-row').forEach(row => {
                row.classList.remove('table-success');
				row.style.background = "";
            });
            const modelRow = document.querySelector(`tr[data-run-id="${runId}"]`);
            if (modelRow) {
                modelRow.classList.add('table-success');
				modelRow.style.background = "rgb(0,150,0)";
            }

			const length = window.networkVisualizer.layers.length
			
			for (var i=0;i < length; i++) {
				removeLayer(i);
			}

			var layers = data.architecture.config.layers;

			var use_reg = false;
			var l2_reg = 0.001;
			var use_batchnorm = false;
			
			for (var i=0;i < layers.length; i++) {
				if (i == layers.length-2) {
					if (layers[layers.length-1].class_name.match("BatchNormalization")) {
						use_batchnorm = true;
						break;
					}
				}

				if (i == layers.length-1) {
					break;
				}

				if (layers[i].class_name.match("Dense")) {
					addLayer('dense', layers[i].config.units, layers[i].config.activation,0.3,0.99);

					if (layers[i].config.kernel_regularizer) {
						if (layers[i].config.kernel_regularizer.class_name === 'L2') {
							use_reg = true;
							l2_reg = layers[i].config.kernel_regularizer.config.l2;
						}
					}
				}

				if (layers[i].class_name.match("Dropout")) {
					addLayer('dropout', 1, 'relu', layers[i].config.rate,0.99);
				}

				if (layers[i].class_name.match("BatchNormalization")) {
					addLayer('batchnorm', 1, 'relu',0.3, layers[i].config.momentum);
				}
			}

			//if (use_batchnorm) {
				document.querySelector("[name=use_batchnorm]").checked = use_batchnorm

				document.querySelector("[name=use_reg]").checked = use_reg
				if (use_reg) {
					document.querySelector("[name=l2_reg]").value = l2_reg
				}
			//}

			var optimizer = data.architecture.compile_config.optimizer.config
			
			var optimizerSelect = document.querySelector("[name=optimizer]")
			
			optimizerSelect.childNodes.forEach(node => {
				node.selected = false
			});

			optimizerSelect.querySelector(`[value=${optimizer.name.toLowerCase()}]`).selected = true
			document.querySelector("[name=learning_rate]").value = optimizer.learning_rate

			var adam = document.getElementById('adam-params')
			var sgd = document.getElementById('sgd-params')

			if (optimizer.name === 'adam') {
				document.querySelector("[name=beta_1]").value = optimizer.beta_1
				document.querySelector("[name=beta_2]").value = optimizer.beta_2

				adam.style.display = "block";
				sgd.style.display = "none";
			}
			else if (optimizer.name === 'sgd') {
				document.querySelector("[name=momentum]").value = optimizer.momentum
				document.querySelector("[name=use_nesterov]").value = optimizer.use_nesterov

				adam.style.display = "none";
				sgd.style.display = "block";
			}

			removeLayer(1);

			document.getElementById('current-model').innerHTML = data.version;
			document.querySelector('[name=current-model-id]').value = data.current;
            
            alert(`Model v${data.version} loaded successfully!`);
        } else {
            alert('Failed to load model: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to load model: ' + error.message);
    })
    .finally(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function deleteModel(event, runId) {
    if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
        return;
    }

	var id = 0;

	if (!document.URL.match('sandbox')) {
		id = event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id.split('neural-newtork')[1];
	}
    
    const btn = constructor.querySelector(`button[onclick="deleteModel('${runId}')"]`);
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Deleting';
    btn.disabled = true;

	//var id = 0;

	//if (document.URL.match('sandbox')) {
	//	id = event.target.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.parentElement.id.split('neural-network')[1];
	//}
    
    fetch(`/api/models/${runId}/delete/`, {
        method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'x-csrftoken': document.querySelector('[name=csrfmiddlewaretoken]').value
		}
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
			var constructor = nodes[id].constructor;

            const row = constructor.querySelector(`tr[data-run-id="${runId}"]`);
            if (row) row.remove();
            
            //$(`#model-select option[value="${runId}"]`).remove();
			constructor.querySelector(`#model-select option[value="${runId}"`).remove();
            
            alert('Model deleted successfully!');
        } else {
            alert('Failed to delete model: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to delete model: ' + error.message);
    })
    .finally(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function compareModels(event) {
	var id = parseInt(event.parentElement.parentElement.parentElement.id.split('neural-network')[1]);
	var constructor = nodes[id].constructor;

    const selectedIds = constructor.querySelector('#model-select').value; // $('#model-select').val();
    if (!selectedIds || selectedIds.length < 2) {
        alert('Please select at least 2 models to compare');
        return;
    }
    
    const compareIds = selectedIds.slice(0, 2);
    
    const btn = constructor.querySelector('#comparison button');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Comparing';
    btn.disabled = true;
    
    $.get(`/api/compare/?run1=${compareIds[0]}&run2=${compareIds[1]}`, function(data) {
        const metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1_score'];
        let tableHtml = '';
        
        metrics.forEach(metric => {
            if (data.metrics[metric]) {
                const m1 = data.metrics[metric].run1;
                const m2 = data.metrics[metric].run2;
                const delta = (m2 - m1).toFixed(4);
                
                tableHtml += `
                    <tr>
                        <td>${metric.charAt(0).toUpperCase() + metric.slice(1)}</td>
                        <td>${m1.toFixed(4)}</td>
                        <td>${m2.toFixed(4)}</td>
                        <td class="${delta > 0 ? 'text-success' : 'text-danger'}">
                            ${delta > 0 ? '+' : ''}${delta}
                        </td>
                    </tr>
                `;
            }
        });

		constructor.querySelector('#metrics-table tbody').innerHTML = tableHtml;
        
        //$('#metrics-table tbody').html(tableHtml);
        
        // Update comparison charts
        updateComparisonChart('compare-acc-chart', 
            ['Accuracy', 'Validation Accuracy'], 
            [data.run1.history.accuracy, data.run1.history.val_accuracy],
            [data.run2.history.accuracy, data.run2.history.val_accuracy], 1
        );
        
        updateComparisonChart('compare-loss-chart', 
            ['Loss', 'Validation Loss'], 
            [data.run1.history.loss, data.run1.history.val_loss],
            [data.run2.history.loss, data.run2.history.val_loss], 2
        );
        
        // Show resource comparison
        //$('#resource-comparison').html(`
		constructor.querySelector('#resource-comparison').innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h5>Model v${data.run1.version}</h5>
                    <p>Training Time: ${data.run1.training_time}s</p>
                    <p>Model Size: ${(data.run1.model_size / 1024).toFixed(2)} KB</p>
                </div>
                <div class="col-md-6">
                    <h5>Model v${data.run2.version}</h5>
                    <p>Training Time: ${data.run2.training_time}s</p>
                    <p>Model Size: ${(data.run2.model_size / 1024).toFixed(2)} KB</p>
                </div>
            </div>
        `;
    })
    .fail(function() {
        alert('Failed to fetch comparison data');
    })
    .always(function() {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}



async function startComplexTraining(event) {
	const id = parseInt(event.parentElement.parentElement.parentElement.id.split('neural-network')[1]);
	const node = nodes[id];
	const data = nodeDataMap.get(node) || {};

	if (!data.csv || data.csv.length === 0) {
		alert('Please upload training data first!');
		return;
	}

	//const constructor = document.querySelector(`#neural-network${id}`);
	const constructor = node.constructor;

	const architecture = node.visualizer.getArchitecture();

	const hyperparams = {
		learning_rate: parseFloat(constructor.querySelector("[name='learning_rate']").value),
		epochs: parseInt(constructor.querySelector("[name='epochs']").value),
		batch_size: parseInt(constructor.querySelector("[name='batch_size']").value),
		optimizer: constructor.querySelector("[name='optimizer']").value,
		
		beta_1: constructor.querySelector("[name='beta_1']").value,
		beta_2: constructor.querySelector("[name='beta_2']").value,

		momentum: constructor.querySelector("[name='momentum']").value,
		nesterov: constructor.querySelector("[name='nesterov']").value,

		use_reg: constructor.querySelector("[name='use_reg']").value,
		l2_reg: constructor.querySelector("[name='l2_reg']").value,
		use_batchnorm: constructor.querySelector("[name='use_batchnorm']").value
	}

	const form = new FormData();
	form.append('data', JSON.stringify({
		input: data.inputColumns,
		output: data.outputColumns,
		mode: 'custom',
		learning_rate: hyperparams.learning_rate,
		epochs: hyperparams.learning_rate,
		batch_size: hyperparams.batch_size,
		optimizer: hyperparams.optimizer,
		beta_1: hyperparams.beta_1,
		beta_2: hyperparams.beta_2,
		momentum: hyperparams.momentum,
		nesterov: hyperparams.nesterov,
		use_reg: hyperparams.use_reg,
		l2_reg: hyperparams.l2_reg,
		use_batchnorm: hyperparams.use_batchnorm,
		architecture: architecture
	}));

	form.append('training_data', constructor.getElementById('training_data').files[0])

	try {
		const response = await fetch(`/api/complex-train`, {
			method: 'POST',
			headers: {
				'x-csrftoken': document.querySelector('[name=csrfmiddlewaretoken]').value
			},
			body: form
		})

		const result = await response.json();
		if (result.success) {
			nodeDataMap.set(id, {
				...data,
				uuid: result.uuid,
			})

			const status = document.querySelector(`[name="trainstatus"] [data-node="${id}"]`);
			status.value = 'true';

			//if (status)

			alert('Model trained successfully!');
		}
		else {
			alert('Trainng failed: ' + result.message);
		}
	}
	catch (error) {
		console.error('Training error: ', error);
		alert('Training failed: ' + error.message);
	}
}

window.startAdvancedTraining = function() { startAdvancedTraining() }

window.startPredicting = function() { startPredicting() }

window.loadModel = function(event, runId) { loadModel(event, runId) }
window.deleteModel = function(event, runId) { deleteModel(event, runId) }
window.compareModels = function(event) { compareModels(event) }



document.addEventListener('DOMContentLoaded', () => {
	if (document.URL.match('sandbox')) {

		initNetworkVisualization()

		var currentRun
		var comparison

		window.removeLayer = function(index) { removeLayer(index) }
	}
	else if (document.URL.match('classify')) {
		const visualizer = new NetworkVisualizer('');
		window.networkVisualizer = visualizer;
		
		visualizer.layers = [
			{ type: 'input', units: 8, label: 'Input Features', dropout: 0.3, batch_norm_momentum: 0.9, weight: 0 },
			{ type: 'dense', units: 128, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.9, weight: 0 },
			{ type: 'dropout', units: 1, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.9, weight: 0 },
			{ type: 'dense', units: 64, activation: 'relu', dropout: 0.3, batch_norm_momentum: 0.9, weight: 0 },
			{ type: 'output', units: 3, activation: 'softmax', dropout: 0.3, batch_norm_momentum: 0.9, weight: 0 },
		]

		/*removeLayer(1,true);
		removeLayer(2,true);
		removeLayer(3,true);
		removeLayer(4,true);

		addLayer('dense', 128, 'relu');
		addLayer('dropout', 1, 'relu', 0.3);
		addLayer('dense', 64, 'relu');*/

		visualizer.render();

		document.querySelector('#classification-results').addEventListener('click', e => {
			document.getElementById('content-container-results').style.display = "block";
		});

		document.querySelector('#close-content-container-results').addEventListener('click', e => {
			document.getElementById('content-container-results').style.display = "none";
		});
	}
	
	
});
