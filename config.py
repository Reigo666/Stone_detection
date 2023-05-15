import os
import yaml
# yaml_path = os.path.join((os.path.dirname(os.path.abspath(__file__))), 'config.yml')
yaml_path = r'D:\jpf\mine_terminal\config.yml'

class MyConfig:
    def __init__(self, path=yaml_path):
        self.config_path = path
        self.config = self.load_data()

    def write2yaml(self):
        """
        存储yaml文件
        """
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.config, f)
        except Exception as err:
            print(err)

    def load_data(self):
        """
        加载yaml文件
        """
        try:
            with open(self.config_path, "r", encoding='utf-8') as f:
                content = f.read()
            yaml_data = yaml.load(content, Loader=yaml.FullLoader)
            return yaml_data
        except Exception as err:
            print(err)


my_config = MyConfig()

if __name__ == '__main__':
    mc = MyConfig(path=yaml_path)
    print(mc.config)
