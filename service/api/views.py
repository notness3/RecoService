import json
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.recommenders.utils import load_model

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
USER_DATABASE = {"admin": {"login": "admin", "password": "admin"}}
MODE = "offline"

if MODE == "online":
    PATH = "./models/user_knn.pickle"
    tfidf_model = load_model(PATH)
    hot_or_warm_users = tfidf_model.users_mapping.keys()
else:
    with open("./models/userknn_model.json", "r", encoding="utf-8") as f:
        user_mapping = json.load(f)
        hot_or_warm_users = user_mapping.keys()

with open("./models/popular_model.json", "r", encoding="utf-8") as f:
    popular_mapping = json.load(f)
    recos_for_cold = popular_mapping[list(popular_mapping.keys())[0]]


with open("./top_100_frequent.json", "r", encoding="utf-8") as f:
    TOP_100 = json.load(f)


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
        user_df = pd.DataFrame([{"user_id": user_id}])

        if user_id in hot_or_warm_users:
            if MODE == "online":
                reco = tfidf_model.predict(user_df, k_recs)["item_id"].to_list()
            else:
                reco = user_mapping[user_id]

        pop_reco = recos_for_cold
        len_recos = k_recs - len(reco)

        reco += [item for item in pop_reco if item not in reco][:len_recos]

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
