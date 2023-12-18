import onnxruntime as ort
import numpy as np

EP_list = [ 'CPUExecutionProvider']

# initialize the model.onnx
# sess = ort.InferenceSession("model.onnx", providers=EP_list,provider_options=ort.SessionOptions())
import onnxruntime as ort
import numpy as np
print("input name")

# Load the ONNX model
session = ort.InferenceSession('model.onnx',providers=EP_list,provider_options=ort.SessionOptions())
print("input output session")
# Assume model has a single input and output
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("input output name")

# Create dummy input data (adjust shape and type according to your model's requirements)
input_data = np.random.randn(1, 3, 224, 224).astype(np.int32)
print("input data")
# Run the model
result = session.run([output_name], {input_name: input_data})
print("result")
# The result is a list of output data (numpy arrays)
print(result)
