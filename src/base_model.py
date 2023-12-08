class BaseModel():
    def __init__(self, dir_):
        self.dir_ = dir_
        self.write_dir = dir_ + "models/last/"
        self.read_dir = dir_ + "models/active/"

    # Abstract
    def load_model(self):
        pass

    # Abstract
    def get_samples(self):
        pass

    # Abstract
    def train(self, train_set, val_set, test_set):
        pass
