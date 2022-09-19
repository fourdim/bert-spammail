import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

from src.index import DetectorIndex

class Detector:
    def __init__(self) -> None:
        self.tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        self.tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
        self.model = self.__build_classifier_model()
        self._model_loaded = False

    def __build_classifier_model(self) -> tf.keras.Model:
        subject_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='subject')
        email_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='email')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        subject_encoder_inputs = preprocessing_layer(subject_input)
        email_encoder_inputs = preprocessing_layer(email_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        subject_outputs = encoder(subject_encoder_inputs)
        subject_net = subject_outputs['pooled_output']
        email_outputs = encoder(email_encoder_inputs)
        email_net = email_outputs['pooled_output']
        net = tf.keras.layers.Concatenate()([subject_net, email_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model([subject_input, email_input], net)

    def load_model(self, path):
        self.model.load_weights(path)
        self._model_loaded = True

    def predict(self, subject, email):
        return DetectorIndex(self.model.predict(x=[np.array([subject,]), np.array([email,])])[0][0])
