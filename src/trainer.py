import pandas as pd
import tensorflow as tf
from official.nlp import optimization
from src.detector import Detector

class DectectorTrainer:
    def __init__(self, train_dataframe) -> None:
        self.ds_train = train_dataframe
        self.ds_train['subject'] = self.ds_train['subject'].astype(str)
        self.ds_train['email'] = self.ds_train['email'].astype(str)

    def train(self, detector: Detector):
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        epochs = 5
        batch_size = 20
        steps_per_epoch = len(self.ds_train) // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = 3e-5
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                num_train_steps=num_train_steps,
                                                num_warmup_steps=num_warmup_steps,
                                                optimizer_type='adamw')
        detector.model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)
        detector.model.fit(x=[self.ds_train['subject'], self.ds_train['email']],
                            y=self.ds_train['spam'],
                            epochs=epochs,
                            batch_size=batch_size,
                            )
        detector._model_loaded = True
