import time

def info(msg):
  print(f'{time.ctime()} {msg}')

info("Importing packages.")
from pathlib import Path
from fastai.vision.all import L, load_learner, PILImage, Resize

# import IPython
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse

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
def predict_img(request: Request, image: UploadFile=File(...)):
  print(f'predict_img')
  normal = Resize(224)
  pil = normal(PILImage.create(image.file))
  # IPython.embed()
  results = learner.predict(pil)

  return {
      "cat": results[0]
    , "vocab": list(learner.dls.vocab)
    , "confidences": results[2].tolist()
  }
