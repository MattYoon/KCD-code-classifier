from re import template
from flask import Flask, render_template
from flask import request
import json
import sys, os
# sys.path.insert(0, '/home/konkuk/KCD-code-classifier/demo/model/')
# sys.path.append('/home/konkuk/KCD-code-classifier/demo/model/')
from ..model.model import infer
import pickle

 # -- coding: utf-8 --
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# load data
with open('demo/pickle/code2cate.pickle', 'rb') as fr:
    code2cate = pickle.load(fr)
with open('demo/pickle/code2name.pickle', 'rb') as fr:
    code2name = pickle.load(fr)

# 뒤에 resource path X, 브라우저 접근
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    data = str(request.json)
    print(data)
    result = infer(data)
    json_data = {str(idx):f"({value}) {code2name[value]} - {code2cate[value]}" for idx, value in enumerate(result)}
    json_data['context'] = data
    print(json_data)
    send = json.dumps(json_data)
    return send

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=False)