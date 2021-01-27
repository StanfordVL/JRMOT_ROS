import tensorflow as tf
from skimage.transform import resize
import numpy as np

class ImageEncoder(object):

    def __init__(self, checkpoint_filename="weights/deep_sort_weights.pb", input_name="images",
                 output_name="features"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x):
        #Resize input to expected size for model
        data_x = resize(data_x[0], self.image_shape, anti_aliasing=True, mode='reflect')
        data_x = np.expand_dims(data_x, 0)
        out = self.session.run(self.output_var, feed_dict={self.input_var: data_x})
        return out

if __name__ == '__main__':
    encoder = ImageEncoder()