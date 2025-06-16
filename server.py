import time

def info(msg):
  print(f'{time.ctime()} {msg}')

# info("Importing packages.")
# from pathlib import Path
# from fastai.vision.all import load_learner

import IPython
from fastapi import FastAPI, Request, UploadFile

api = FastAPI()

@api.get('/ok')
def ok_200():
    return "ok"


@api.post('/predict')
def predict_img(request: Request):
  print(f'predict_img')
  IPython.embed()
  return 200
