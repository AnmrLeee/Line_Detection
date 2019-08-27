from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from fully_conv_NN import model
from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes='Ture', show_layer_names='Ture', rankdir='TB')

# model即为要可视化的网络模型
SVG(model_to_dot(model).create(prog='dot', format='svg'))

