from flask import Flask, render_template
from flask_cors import CORS
import logging

# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/index')
def home():
    logger.info("Rendering index.html")
    return render_template('index.html')

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)