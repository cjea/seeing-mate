import time
from pathlib import Path
from fastai.vision.all import L, load_learner, PILImage, Resize
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse

def info(msg):
  return None
  print(f'{time.ctime()} {msg}')

api = FastAPI()
learner = load_learner("model.pkl")

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
def predict_img(image: UploadFile=File(...)):
  pil = PILImage.create(image.file)
  normal = Resize(224)
  results = learner.predict(normal(pil))

  return {
      "cat": results[0]
    , "vocab": list(learner.dls.vocab)
    , "confidences": results[2].tolist()
  }
