from Model import Model
class SegnetModel(Model):
    def add_placeholders(self):
        pass

    def create_feed_dict(self):
        pass

    def add_prediction_op(self):
        pass

    def add_loss_op(self):
        pass

    def add_training_op(self):
        pass

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        pass

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
