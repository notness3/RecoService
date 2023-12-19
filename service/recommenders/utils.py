import json
import os
from pickle import Unpickler

from service.recommenders.userknn import UserKnn


class CustomUnpickler(Unpickler):
    def find_class(self, module, name):
        if name == "UserKnn":
            return UserKnn
        return super().find_class(module, name)


def load_model(path: str):
    with open(os.path.join(path), "rb") as f:
        return CustomUnpickler(f).load()


def get_recos(main_model: dict, popular_model: list, user_id: str, k_recs: int = 10):
    if user_id in main_model:
        return main_model[user_id]
    return popular_model[:k_recs]


def load_json_model(path: str):
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    return model
