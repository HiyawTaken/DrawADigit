import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

def activation_function(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def load_network(filename="mnist_network_weights.npz"):
    data = np.load(filename)

    weight_keys = sorted(
        [key for key in data.files if key.startswith("weight_")],
        key=lambda key: int(key.split("_")[1])
    )

    bias_keys = sorted(
        [key for key in data.files if key.startswith("bias_")],
        key=lambda key: int(key.split("_")[1])
    )

    if len(weight_keys) == 0:
        raise ValueError("No weights found in saved network file.")

    if len(weight_keys) != len(bias_keys):
        raise ValueError("Weight and bias count do not match.")

    sizes = [data[weight_keys[0]].shape[1]]

    for key in weight_keys:
        sizes.append(data[key].shape[0])

    network = Network(sizes)

    for i, key in enumerate(weight_keys):
        network.weights[i] = np.array(data[key])

    for i, key in enumerate(bias_keys):
        network.biases[i] = np.array(data[key])

    return network


def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img).astype(np.float32) / 255.0

    img_array = img_array.reshape(784, 1)

    return np.array(img_array)


network = load_network("mnist_network_weights.npz")


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/frontend.html")

def forward_pass(network, inputs):
    outputs = []
    z_values = []
    curr_inputs = inputs

    for w, b in list(zip(network.weights, network.biases))[:-1]:
        z = np.dot(w, curr_inputs) + b
        z_values.append(z)
        output = activation_function(z)
        outputs.append(output)
        curr_inputs = output

    w, b = network.weights[-1], network.biases[-1]
    z = np.dot(w, curr_inputs) + b
    z_values.append(z)
    output = softmax(z)
    outputs.append(output)

    return outputs, z_values


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        x = preprocess_image(img)

        outputs, _ = forward_pass(network, x)

        probabilities = outputs[-1]
        prediction = int(np.argmax(probabilities, axis=0)[0])
        confidence = float(np.max(probabilities))

        return {
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))