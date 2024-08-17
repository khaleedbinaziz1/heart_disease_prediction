from django.shortcuts import render 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 

from predictor.forms import HeartDiseaseForm 


def heart(request): 
	# Read the heart disease training data from a CSV file 
	df = pd.read_csv('static/Heart_train.csv') 
	data = df.values 
	X = data[:, :-1] # Input features (all columns except the last one) 
	Y = data[:, -1:] # Target variable (last column) 


	value = '' 

	if request.method == 'POST': 
		# Retrieve the user input from the form 
		age = float(request.POST['age']) 
		sex = float(request.POST['sex']) 
		cp = float(request.POST['cp']) 
		trestbps = float(request.POST['trestbps']) 
		chol = float(request.POST['chol']) 
		fbs = float(request.POST['fbs']) 
		restecg = float(request.POST['restecg']) 
		thalach = float(request.POST['thalach']) 
		exang = float(request.POST['exang']) 
		oldpeak = float(request.POST['oldpeak']) 
		slope = float(request.POST['slope']) 
		ca = float(request.POST['ca']) 
		thal = float(request.POST['thal']) 

		# Create a numpy array with the user's data 
		user_data = np.array( 
			(age, 
			sex, 
			cp, 
			trestbps, 
			chol, 
			fbs, 
			restecg, 
			thalach, 
			exang, 
			oldpeak, 
			slope, 
			ca, 
			thal) 
		).reshape(1, 13) 

		# Create and train a Random Forest Classifier model 
		rf = RandomForestClassifier( 
			n_estimators=16, 
			criterion='entropy', 
			max_depth=9
		) 

		rf.fit(np.nan_to_num(X), Y) # Train the model using the training data 
		rf.score(np.nan_to_num(X), Y) # Evaluate the model's accuracy 
		predictions = rf.predict(user_data) # Make predictions on the user's data 

		if int(predictions[0]) == 1: 
			value = 'have' # User is predicted to have heart disease 
		elif int(predictions[0]) == 0: 
			value = "don\'t have" # User is predicted to not have heart disease 

	return render(request, 
				'heart.html', 
				{ 
					'context': value, 
					'title': 'Heart Disease Prediction', 
					'active': 'btn btn-success peach-gradient text-white', 
					'heart': True, 
					'form': HeartDiseaseForm(), 
				}) 


def home(request): 
	return render(request, 
				'home.html') 
