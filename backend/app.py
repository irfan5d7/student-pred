from flask import Flask, request, jsonify
import pandas as pd
from predicting_students_admission import model_processing
from flask_cors import CORS, cross_origin
from templates.tools import updateDb, selectDb
import io
from base64 import encodebytes
from PIL import Image

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class User:
    def __int__(self, u_name, password):
        self.user_name = u_name
        self.password = password

def register(user):
    updateDb(user)

def logIn(user):
    selectDb(user)


def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


def prediction(vals, selected_model):
    new_pred_vals = vals[:2] + [0] * 4
    new_pred_vals[vals[2] + 1] = 1
    df_pred = pd.DataFrame(columns=['gre', 'gpa', 'rank_1', 'rank_2', 'rank_3', 'rank_4'])
    df_pred.loc[0] = new_pred_vals
    print(selected_model)
    result = selected_model.predict(df_pred)
    return result[0]


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    gre = request.form.get("gre", type=int, default=0)
    gpa = request.form.get("gpa", type=float, default=0)
    rank = request.form.get("rank", type=int, default=0)
    model_sel = request.form.get('select_model')
    all_models = model_processing()
    pred = prediction([gre, gpa, rank], all_models[model_sel])
    resp = {1: "Admit", 0: "Reject"}
    plot_64 = get_response_image("foo.png")
    return jsonify(
        result=resp[pred],
        img=plot_64
    )


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
