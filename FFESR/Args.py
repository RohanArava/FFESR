import os
import yaml
class Args:
    def __init__(self, config_path, data_path, test_path, save_path):
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded')
        self.model = self.config['model']
        self.path = os.path.join(data_path, "HR")
        self.test_path = os.path.join(test_path, "X2", "HR")
        self.save_path = os.path.join(save_path, 'Models')
        os.makedirs(self.save_path, exist_ok=True)
        self.plot_path = os.path.join(save_path, 'Plots', "SsimVsCompressionRate.png")
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        
args = Args(
    config_path="ITSRN/code/configs/train/train_itnsr.yaml", 
    data_path="data/Train", 
    test_path="data/Test",
    save_path="data/Results"
    )