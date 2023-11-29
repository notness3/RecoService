import os
from pickle import Unpickler


class CustomUnpickler(Unpickler):
    def find_class(self, module, name):
        if name == "UserKnn":
            from service.recommenders.userknn import UserKnn

            return UserKnn
        return super().find_class(module, name)


def load_model(path: str):
    with open(os.path.join(path), "rb") as f:
        return CustomUnpickler(f).load()
