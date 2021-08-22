import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
class IntentClassifier:
    def __init__(self,classes,model,tokenizer,label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred,1))[0]
    
    def get_probability(self,text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        self.probability_result = dict()
        for idx, prediction in enumerate(self.pred[0]):
            self.probability_result[self.classes[idx]] = prediction
        return self.probability_result


import pickle

from tensorflow.python.keras.models import load_model
model = load_model('models/intents.h5')

with open('utils/classes.pkl','rb') as file:
  classes = pickle.load(file)

with open('utils/tokenizer.pkl','rb') as file:
  tokenizer = pickle.load(file)

with open('utils/label_encoder.pkl','rb') as file:
  label_encoder = pickle.load(file)


nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
prob = nlu.get_probability("ocupo que apagues la impresora")



# print(nlu.get_probability("prende la impresora"))
# print(nlu.get_probability("porfa pausa la impresora"))
# print(nlu.get_probability("continúa la impresión"))
# print(nlu.get_probability("a qué temperatura estamos"))