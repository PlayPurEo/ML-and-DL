# author : 'wangzhong';
# date: 15/01/2021 19:50

import importlib
from models.base_model import BaseModel

if __name__ == '__main__':
    model_filename = "models." + "cycle_gan" + "_model"
    modellib = importlib.import_module(model_filename)
    print(modellib)

    # print(modellib.__dict__.items())
    for name, cls in modellib.__dict__.items():
        print(name)
        print(cls)