import json
import os
import pickle
from typing import List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.recommenders.utils import get_recos, load_json_model

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
USER_DATABASE = {"admin": {"login": "admin", "password": "admin"}}
MODE = "offline"


with open("./models/userknn_model.json", "r", encoding="utf-8") as f:
    user_mapping = json.load(f)
    hot_or_warm_users = user_mapping.keys()

if os.path.exists("./models/ann_lightfm.pickle"):
    with open("./models/ann_lightfm.pickle", "rb") as file:
        ann_lightfm = pickle.load(file)

with open("./top_100_frequent.json", "r", encoding="utf-8") as f:
    TOP_100 = json.load(f)


model_dict = {
    "autoencoder": load_json_model("./models/autoencoder.json"),
    "dssm": load_json_model("./models/dssm.json"),
    "recbone": load_json_model("./models/recbone.json"),
    "als+lightgbm": load_json_model("./models/2stage.json"),
}


class LoginData(BaseModel):
    access_token: str
    token_type: str


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


def get_current_user(token: str = Depends(oauth2_scheme)):
    user = USER_DATABASE.get(token)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.get(path="/health", tags=["Health"], responses={401: {"description": "Authentication error"}})
async def health(token: str = Depends(get_current_user)) -> str:
    app_logger.info(f"Request token: {token}")
    return "I am alive"


# flake8: noqa: C901
@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={401: {"description": "Authentication error"}, 404: {"description": "Wrong model_name or user_id"}},
)
async def get_reco(
    request: Request, model_name: str, user_id: int, token: str = Depends(get_current_user)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}, token: {token}")
    k_recs = request.app.state.k_recs
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in request.app.state.available_models:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")
    if model_name == "top_frequent":
        reco = list(TOP_100.values())[:k_recs]

    if model_name == "tfidf":
        reco = []

        if user_id in hot_or_warm_users:
            reco = user_mapping[user_id]

        pop_reco = list(TOP_100.values())[:k_recs]
        len_recos = k_recs - len(reco)

        reco += [item for item in pop_reco if item not in reco][:len_recos]

    if model_name == "ann_lightfm":
        if user_id in ann_lightfm.user_id_map.external_ids:
            reco = ann_lightfm.get_item_list_for_user(user_id, top_n=k_recs).tolist()
        else:
            reco = list(TOP_100.values())[:k_recs]

    if model_name in model_dict:
        userid = str(user_id)
        reco = get_recos(model_dict[model_name], list(TOP_100.values()), userid)


    return RecoResponse(user_id=user_id, items=reco)


@router.post("/login", tags=["Login"], responses={400: {"description": "Wrong login or password"}})
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> LoginData:
    user_record = USER_DATABASE.get(form_data.username)
    if not user_record:
        raise HTTPException(status_code=400, detail="Wrong login")

    if not form_data.password == user_record["password"]:
        raise HTTPException(status_code=400, detail="Wrong password")

    return LoginData(access_token=user_record["login"], token_type="bearer")


def add_views(app: FastAPI) -> None:
    app.include_router(router)
