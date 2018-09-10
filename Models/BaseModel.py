import tensorflow as tf
class BaseModel:

    #Set up default parameters that are used by all trainers
    #Standardizes model creation
    def __init(self):
        self.input_ph = None
        self.label_ph = None
        self.loss_fn = None
        self.minimize_op = None
