import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model

def init():
    global model
    from sklearn.externals import joblib
    model_path = Model.get_model_path(model_name = 'mymodel')
    model = joblib.load(model_path)

def run(rawdata):
    try:
        data = json.loads(rawdata)['data']
        data = numpy.array(data).reshape(1,len(data))
        result = model.predict(data)[0]
    except Exception as e:
        result = str(e)
    return json.dumps({"result":result})