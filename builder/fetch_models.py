from concurrent.futures import ThreadPoolExecutor
from stable_whisper import modify_model
import whisper

model_names = ["base"]
selected_model = "base"

def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(3):
        while True:
            try:
                loaded_model = whisper.load_model(
                    selected_model)
            except AttributeError:
                continue

            break

    return selected_model, loaded_model


models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model
