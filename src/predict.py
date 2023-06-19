
# import torch
import numpy as np
import json
from runpod.serverless.utils import rp_cuda

from stable_whisper import modify_model

model = whisper.load_model("base")
modify_model(model)

result = model.transcribe(audio, language="en", mel_first=True, demucs=True, vad=True)
result.save_as_json('audio.json')

with open('audio.json', 'r') as file:
    # Load the JSON data into a variable
    result = json.load(file)


    return result
