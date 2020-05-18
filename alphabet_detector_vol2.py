from keras import models
from keras.models import load_model
from keras.models import model_from_json, load_model
import glob

#_______ LOADING THE TRAINED MODEL TO MAKE PREDICTIONS ON UNSEEN DATA ____
model = models.load_model('my_model.h5')

#______ OBJECT OPENING TRAINED MODEL AND PREDICTING IMAGES FROM CAPTURED DATA_____

class SignLanguageModel(object):

    LETTER_LIST = ["A", "B", "C", "D", "E","F" ]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_letter(self, img):
        self.preds = self.loaded_model.predict(img)
        print (self.preds)
        return SignLanguageModel.LETTER_LIST[np.argmax(self.preds)]





#Creating the images for recognition

images=glob.glob("Desktop/Code/ImageRecognition/datacapture/*.png")
print imaggg
images_for_recognition = []

for image in images:
    img = Image.open(image)
    images_for_recognition.append(img)
    display(img)


#___MAKING THE PREDICTION ON THE CAPTURED DATA _____
model_json_file =  "/Users/ewa_anna_szyszka/Desktop/model.json"
model_weights_file = "/Users/ewa_anna_szyszka/Desktop/my_model.h5"

'''Setting up the '''
for i in images_for_recognition:
    new_array = cv2.resize(np.array(i), (50, 50))
    new_array = new_array.reshape(1,50,50,3)
    a = SignLanguageModel(model_json_file, model_weights_file)
    print(a.predict_letter(new_array))
