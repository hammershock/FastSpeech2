import yaml


def load_config(config_path):
    return yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
