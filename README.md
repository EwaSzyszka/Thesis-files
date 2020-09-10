# Sign Language recognition


This folder contains implementation of a hand gesture recogniser for polish sign language. Often refered to in literature as PJM. 
As of know this readme contains solely short descriptions of what each script does. This is subject to change in the future, as the project evolves.


## Fingerdetection.py

This is the main detection program implemented in python. <br/>
It has the following functionalities:

(a) skin color detection <br/>
(b) convex hull and centre point of the hand <br/>
(c) Type of hand movement clasifier [eg. 'Middle_vertical_movement','Middle_horizontal_movement' ] <br/>
(d) Press a key trigger to run the CNN classification 

## ImageClassifier.mlmodel

This is a trained model for the classifier. As project evolves this model will be replaced by a more acurate one. 

## alphabet_detector.ipynb

This is a full VGG model trained (line [21]) and a  test of the PJM alphabet classifier. 
The code still contains some of early failed attempts of VGG-16 implementation for purpose of transparency regarding to which solutions were tried so far.

##  alphabet_detector.py

Very similar document to the file above. The difference is that here I gave a shoot to a simplified VGG-16 model

## alphabet_detector_vol2.py 

Test of functionalities such as:

(a) Loading a model - my_model.h5
(b) Saving key frames for recognition

## dcgan.py

First attempt at DCGANs, a final DCGAN generated images can be found here:
https://colab.research.google.com/drive/16gjTCVdeepDHS61SkQk02ndLNDC9syoR?usp=sharing

## expeiment_HV.py

Testing hand movement clasifier and setting the centroids values, which will be the thresholds for types of movements detected.  <br/>
[eg. 'Middle_vertical_movement','Middle_horizontal_movement' ] <br/>

## key_frames.py

Savig the key frames, which will be later used by the CNN to classify the gesture.
