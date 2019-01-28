import tensorflow as tf
from abc import abstractmethod
class Model(object):
    @abstractmethod
    def add_placeholders(self):
        pass

    @abstractmethod
    def create_feed_dict(self):
        pass

    @abstractmethod
    def add_prediction_op(self):
        pass

    @abstractmethod
    def add_loss_op(self, pred):
        pass

    @abstractmethod
    def add_training_op(self, loss):
        pass

    @abstractmethod
    def train_on_batch(self, sess, inputs_batch, labels_batch):
        pass

    @abstractmethod
    def predict_on_batch(self, sess, inputs_batch):
        pass

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

