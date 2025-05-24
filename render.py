from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from model.aac_utils import aac_feature
from model.dpc_utils import dpc_feature
from model.pssm_utils import pssm_feature
import logging
from flask_cors import CORS

# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

aac_model = load_model('model/aac_cnn_model.keras')
dpc_model = load_model('model/dpc_cnn_model.keras')
pssm_model = load_model('model/pssm_cnn_model.keras')

@app.route('/')
@app.route('/index')
def home():
    logger.info("Rendering index.html")
    return render_template('index.html')

@app.route('/aac_page')
def aac_page():
    logger.info("Rendering AAC_page.html")
    return render_template('AAC_page.html')

@app.route('/dpc_page')
def dpc_page():
    logger.info("Rendering DPCpage.html")
    return render_template('DPC_page.html')

@app.route('/pssm_page')
def pssm_page():
    logger.info("Rendering PSSM_page.html")
    return render_template('PSSM_page.html')

@app.route('/aac_predict', methods=['POST'])
def aac_predict():
    data = request.get_json()
    sequence = data.get('sequence', '')
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    # 轉換成 AAC 特徵
    aac_vec = aac_feature(sequence)
    # CNN 可能需要 reshape
    aac_vec = aac_vec.reshape(1, 20, 1)  # 根據你的模型 input shape 調整

    # 預測
    pred = aac_model.predict(aac_vec)
    result = int(np.argmax(pred, axis=1)[0])
    probability = float(pred[0][result])  # 預測類別的機率

    # 如果你想同時回傳兩個類別的機率
    probability_0 = float(pred[0][0])
    probability_1 = float(pred[0][1])

    return jsonify({
        'prediction': result,
        'probability': probability,
        'probability_0': probability_0,
        'probability_1': probability_1
    })

@app.route('/dpc_predict', methods=['POST'])
def dpc_predict():
    data = request.get_json()
    sequence = data.get('sequence', '')
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    # 轉換成 DPC 特徵
    dpc_vec = dpc_feature(sequence)  # shape: (400,)
    dpc_vec = dpc_vec.reshape(1, 400, 1)  # 根據你的模型 input shape 調整

    # 預測
    pred = dpc_model.predict(dpc_vec)
    result = int(np.argmax(pred, axis=1)[0])
    probability = float(pred[0][result])
    probability_0 = float(pred[0][0])
    probability_1 = float(pred[0][1])

    return jsonify({
        'prediction': result,
        'probability': probability,
        'probability_0': probability_0,
        'probability_1': probability_1
    })

@app.route('/pssm_predict', methods=['POST'])
def pssm_predict():
    data = request.get_json()
    sequence = data.get('sequence', '')
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    # 轉換成 PSSM 特徵
    pssm_vec = pssm_feature(sequence)  
    pssm_vec = pssm_vec.reshape(1, 20, 20, 1)  # CNN input shape

    # 預測
    pred = pssm_model.predict(pssm_vec)
    result = int(np.argmax(pred, axis=1)[0])
    probability = float(pred[0][result])
    probability_0 = float(pred[0][0])
    probability_1 = float(pred[0][1])

    return jsonify({
        'prediction': result,
        'probability': probability,
        'probability_0': probability_0,
        'probability_1': probability_1
    })

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)