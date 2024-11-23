import yaml
class Args:
    def __init__(self, config_path, data_path):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            # print('config loaded:', self.config)
        self.model = self.config['model']
        self.path = data_path
        
args = Args("../ITSRN/code/configs/train/train_itnsr.yaml", )