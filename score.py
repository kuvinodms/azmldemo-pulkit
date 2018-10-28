def init():
    from sklearn.externals import joblib
    from azureml.core.model import Model

    global model
    model_path = Model.get_model_path('ucimodel')
    print("model_path: " + model_path)
    model = joblib.load(model_path)

def run(input_df):
    import json
    import pandas as pd

    df = pd.DataFrame(json.loads(input_df)["input_df"],columns=['height', 'width', 'shoe_size'])
    print ("jsoninput2:\n" + df.to_string())
    pred = model.predict(df)
    return json.dumps(str(pred[0]))

    #pred = model.predict(input_df)
    #return json.dumps(str(pred[0]))

def main():
  from azureml.api.schema.dataTypes import DataTypes
  from azureml.api.schema.sampleDefinition import SampleDefinition
  from azureml.api.realtime.services import generate_schema
  import pandas

  df = pandas.DataFrame(data=[[190, 60, 38]], columns=['height', 'width', 'shoe_size'])

  # Test the functions' output
  init()
  input2 = '{"input_df": [{"width": 60, "shoe_size": 38, "height": 190}]}'
  input1 = pandas.DataFrame([[190, 60, 38]])
  print("input1df:\n" + input1.to_string())
  print("Result: " + run(input2))

  inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}

  # Generate the service_schema.json
  #generate_schema(run_func=run, inputs=inputs, filepath='service_schema.json')
  print("Schema generated")

if __name__ == "__main__":
    main()