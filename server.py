import time

def info(msg):
  print(f'{time.ctime()} {msg}')

# info("Importing packages.")
# from pathlib import Path
# from fastai.vision.all import load_learner

import IPython
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse

api = FastAPI()

@api.get('/ok')
def ok_200():
    return "ok"

@api.get('/index.html')
def index_html():
  return FileResponse("index.html")

@api.get('/style.css')
def style_css():
  return FileResponse("style.css")


@api.post('/predict')
def predict_img(request: Request, image: UploadFile=File(...)):
  print(f'predict_img')
  # IPython.embed()
  return 200
