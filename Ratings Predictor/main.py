# Importing the libraries
from screens.main_gui import GraphicalUserInterface
from models.training_model import NeuralEngine

# Testing the model
review = "Guests liked the clean, updated rooms, though some said they were small & maintenance could be improved · Rooms had views · Some guests said the bathrooms were small & cleanliness could be improved"
engine = NeuralEngine()
engine.classify_train()
engine.predict_classifier(review)
