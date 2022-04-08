import numpy as np
import json
import tensorflow as tf
import pandas as pd
import joblib
from PIL import Image

from flask import Flask
from flask import request
import tensorflow_text as text

app = Flask(__name__)

@app.route("/", methods=['POST'])
def detectObject():

    descriptions_location = request.form['descriptions_location']
    df = pd.read_csv(descriptions_location)

    model_location = request.form['model_location']
    file = request.files['img']
    img = Image.open(file.stream)

    newsize = (248, 248)
    img = img.resize(newsize)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # tf.saved_model.LoadOptions = '/job:localhost'
    cnn_from_joblib = tf.keras.models.load_model(model_location)
    #joblib.load("filename.pkl")
    predictions = cnn_from_joblib.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_name_list = [
        'Alpro-Blueberry-Soyghurt', 'Alpro-Fresh-Soy-Milk',
        'Alpro-Shelf-Soy-Milk', 'Alpro-Vanilla-Soyghurt', 'Anjou', 'Asparagus',
        'Aubergine', 'Avocado', 'Banana', 'Beef-Tomato', 'Bravo-Apple-Juice',
        'Bravo-Orange-Juice', 'Brown-Cap-Mushroom', 'Cabbage', 'Cantaloupe',
        'Carrots', 'Conference', 'Cucumber', 'Floury-Potato', 'Galia-Melon',
        'Garlic', 'Ginger', 'God-Morgon-Apple-Juice',
        'God-Morgon-Orange-Juice', 'God-Morgon-Orange-Red-Grapefruit-Juice',
        'God-Morgon-Red-Grapefruit-Juice', 'Golden-Delicious', 'Granny-Smith',
        'Green-Bell-Pepper', 'Honeydew-Melon', 'Kaiser', 'Kiwi', 'Leek',
        'Lemon', 'Lime', 'Mango', 'Nectarine', 'Oatly-Natural-Oatghurt',
        'Oatly-Oat-Milk', 'Orange', 'Orange-Bell-Pepper', 'Papaya',
        'Passion-Fruit', 'Peach', 'Pineapple', 'Pink-Lady', 'Plum',
        'Pomegranate', 'Red-Beet', 'Red-Bell-Pepper', 'Red-Delicious',
        'Red-Grapefruit', 'Regular-Tomato', 'Royal-Gala', 'Satsumas',
        'Solid-Potato', 'Sweet-Potato', 'Tropicana-Apple-Juice',
        'Tropicana-Golden-Grapefruit', 'Tropicana-Juice-Smooth',
        'Tropicana-Mandarin-Morning', 'Valio-Vanilla-Yoghurt', 'Vine-Tomato',
        'Watermelon', 'Yellow-Bell-Pepper', 'Yellow-Onion',
        'Yoggi-Strawberry-Yoghurt', 'Yoggi-Vanilla-Yoghurt', 'Zucchini'
    ]

    class_name = class_name_list[np.argmax(score)]
    desc = df.loc[df['category'] == class_name]['description'].to_list()
    
    json_file = {}
    json_file['name'] = class_name
    json_file['description'] = desc[0]

    json_object = json.dumps(json_file, indent=4)
    return json_object