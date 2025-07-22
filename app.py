from flask import Flask, request, render_template
import numpy as np
import warnings
from keras.models import load_model
from feature import featureExtraction

warnings.filterwarnings('ignore')

app = Flask(__name__)

autoencoder = load_model("models/autoencoder_model.h5")
RECONSTRUCTION_ERROR_THRESHOLD = 0.05

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        try:
            features = featureExtraction(url, label=0)
            x = np.array(features[:-1]).reshape(1, -1)

            reconstructed = autoencoder.predict(x)
            reconstruction_error = np.mean(np.abs(x - reconstructed))

            prob_safe = max(0.0, min(1.0, 1 - reconstruction_error))
            return render_template("index.html", xx=round(prob_safe, 2), url=url)
        except Exception as e:
            return render_template("index.html", xx=-1, url="Error: " + str(e))
    
    return render_template("index.html", xx=-1)

if __name__ == '__main__':
    app.run(debug=True)