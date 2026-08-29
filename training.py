import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

import numpy as np

# Run this to train the model. TOI and Cumulative .csv files are combined.
def train_and_save_model():
	toi = pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv")
	cumulative = pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv")

	cumulative['tfopwg_disp'] = 'CP'

	combined = pd.concat([toi,cumulative], ignore_index=True)
	#combined = combined[combined['disposition'].notna()]
	features = ['pl_orbper','pl_trandur','pl_rade','pl_bmasse','st_teff','st_rad','st_mass','sy_dist']
	target = "tfopwg_disp"

	print("training")

	df_clean = combined.dropna(subset=features+[target])
	X = df_clean[features]
	y = df_clean[target].map({'CP':0, 'PC':1,'FP':2})
	# APC = ambiguous planetary candidate
	# CP = confirmed
	# FA = false alarm
	# FP = false positive
	# KP = known planet
	# PC = planetary candidate

	# additional note from dev: this neural network architecture is actually bad. come up with your own for better accurate results.
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(128,activation='relu',input_shape=(len(features),)),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(64,activation='relu'),
		tf.keras.layers.Dropout(0.3),#
		tf.keras.layers.Dense(64,activation='relu'),#
		tf.keras.layers.Dense(3,activation='softmax')
	])

	model.compile(optimizer='adam',
			   loss='sparse_categorical_crossentropy',
			   metrics=['accuracy'])
	model.fit(X_scaled, y, epochs=50,batch_size=32,validation_split=0.2,verbose=1)

	model.save('exoplanet_model.h5')
	joblib.dump(scaler, 'scaler.pkl')
	print("Model and scaler saved successfully")

# You can run this to combine TOI and Cumulative .csv files into one testing.csv file.
def new_csv():
	toi = pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+toi&format=csv")
	cumulative = pd.read_csv("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv")

	cumulative['tfopwg_disp'] = 'CONFIRMED'

	combined = pd.concat([toi,cumulative], ignore_index=True)

	#half_point = len(combined) // 2
	df_first_half = combined#.iloc[:half_point]
	df_second_half = combined#.iloc[half_point:]
	df_first_half.to_csv("C:\\Users\\Elias\\Downloads\\training.csv", index=False)
	df_second_half.to_csv("testing.csv", index=False)

train_and_save_model()
