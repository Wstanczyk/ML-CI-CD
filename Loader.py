import numpy as np
import joblib

model = joblib.load("model_v1.joblib")

#Przykładowy input, ze względu że korzystamy z wbudowanego datasetu - Iris
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  #na bazie cech kwiatu


prediction = model.predict(sample_input)
predicted_class = np.argmax(prediction)  #Klasa o najwyższym prawdopodobieństwie
if predicted_class == 0:
    print('To Setosa (0)')
elif predicted_class == 1:
    print('To Versicolor (1)')
elif predicted_class == 2:
    print('To Virginica (2)')


