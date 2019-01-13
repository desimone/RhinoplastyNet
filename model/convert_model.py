from __future__ import absolute_import, division, print_function
import keras
import coremltools
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model

with CustomObjectScope({
        'relu6': keras.applications.mobilenet.relu6,
        'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
}):
    model = load_model('./models/MobileNetV2.hdf5')

print("keras : {}".format(keras.__version__))

model = coremltools.converters.keras.convert(model, input_names=['image'], image_input_names='image')
model.author = 'Bobby D. DeSimone'
model.license = 'MIT'
model.short_description = '.'
model.input_description['image'] = 'A 224x224 image'
model.output_description['output1'] = ' '
model.save('85_mobilenetv2_rhinoplasty.mlmodel')
print('model converted')
