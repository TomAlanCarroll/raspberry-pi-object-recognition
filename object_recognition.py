import urllib
import zipfile
import numpy as np
import mxnet as mx
import time
import cv2


def unzip(zip_file, outdir):
    """
    Unzip a given 'zip_file' into the output directory 'outdir'.
    """
    zf = zipfile.ZipFile(zip_file, "r")
    zf.extractall(outdir)


# Load the network parameters from the cloud
def load_model(model_url='http://data.mxnet.io/models/imagenet/squeezenet'):
    urllib.urlretrieve(model_url, 'model.zip')
    unzip('model.zip', 'model')
    sym, arg_params, aux_params = mx.model.load_checkpoint('Image_Model', 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    mod.set_params(arg_params, aux_params)
    return mod


# Predict on an Image
def predict(image_path, mod):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    mod.forward(Batch([mx.nd.array(img)]))
    return mod.get_outputs()[0].asnumpy()


camera_port = 0
camera = cv2.VideoCapture(camera_port)
time.sleep(0.5)  # If you don't wait, the image will be dark
return_value, image = camera.read()
cv2.imwrite('to_detect.png', image)
del (camera)  # so that others can use the camera as soon as possible
