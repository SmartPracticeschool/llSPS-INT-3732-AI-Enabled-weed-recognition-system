from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import numpy as np
from keras.preprocessing import image 


from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model
from keras import backend
from tensorflow.keras import backend

import tensorflow as tf

global graph
tf.compat.v1.disable_eager_execution()
graph=tf.compat.v1.get_default_graph()

#global graph
#graph = tf.get_default_graph()


from skimage.transform import resize

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = 'models/weed.h5'

# Load your trained model
model =tf.keras.models.load_model(model_path)
       # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
        index = {'Hedge bindweed':"Hedge bindweed, Calystegia sepium, is able to spread rapidly to creep between cultivated plants, making it difficult to eradicate. It’s able to re-grow from small pieces of cream-white root, so cultivating a border often aids its spread. It can make large clumps of foliage, obscuring and smothering small plants. Seed is produced following the cream-white trumpet flowers, which also allows this weed to spread.Another popular method for controlling bindweed is to prune the vines back to the ground repeatedly, whenever they appear.",
                 'Grass':"Crabgrass is an annual weed whose seeds germinate during spring and summer.Crabgrass favors sandy, compacted soil where the grass is in a weakened, thinned condition. Bare, thin lawns allow sunlight to directly hit the soil, which is an ideal condition for crabgrass germination. In addition, weeds like crabgrass like it when your lawn has excess water and phosphorus levels.You can apply a pre-emergent fertilizer before the germination period to organically control crabgrass.Weeds are better adapted to adverse growing conditions than most lawn grasses. Shallow, frequent watering encourages shallow root growth, making the grass more likely to suffer during periods of heat and drought. That kind of stress can lead to thin patches and bare spots that crabgrass will take advantage of.",
                 'Stinging Nettle':"Burning and stinging nettles can be controlled by removing plants by hand. However, it is important to wear gloves to protect skin from the stinging hairs. For stinging nettle, ensure that the underground portion called rhizomes are removed or the plants will regrow. Because stinging nettles are native to California and the western United States, control should only be performed in areas where they cause economic or health problems. Close mowing can prevent the development of fruit. Be aware that cultivating the soil may spread the rhizomes of stinging nettle, thus increasing the size of the population. Repeated cultivation works best as a control for this weed."
                 }
        
        if preds[0]==0:
            text= index['Hedge bindweed']
            t="HEDGE BINDWEED    :" + text
        if preds[0]==1:
            text=index['Grass']
            t="GRASS    :" +text
        if preds[0]==2:
            text=index['Stinging Nettle']
            t="STINGING NETTLE   :"+text
        
               # ImageNet Decode
        

        
        return t
    


if __name__ == '__main__':
    app.run(debug=False,threaded = False)


