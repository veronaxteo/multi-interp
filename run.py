#from pycallcc import *
import numpy as np

#if __name__ != "__main__":
if True:
    import base64
    from io import BytesIO
    from PIL import Image
    
    import argparse
    import asyncio
    import json
    import time
    import threading
    import uuid
    import requests
    import torch
    import uvicorn
    import transformers
    from PIL import Image
    
    from fastapi import FastAPI, Request, BackgroundTasks
    from fastapi.responses import StreamingResponse
    from functools import partial
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    from bunny.constants import WORKER_HEART_BEAT_INTERVAL
    from bunny.util.utils import (build_logger, server_error_msg, pretty_print_semaphore)
    from bunny.model.builder import load_pretrained_model
    from bunny.util.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, get_model_name_from_path, \
        KeywordsStoppingCriteria
    from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    
class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name, model_type,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            self.model_name = get_model_name_from_path(model_path)
        else:
            self.model_name = model_name

        self.device = device
        print(f"Loading the model {self.model_name} on worker {worker_id} ...")
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, model_type, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = True


if 1:
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=model.dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=model.dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False


        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.",
                        "error_code": 0}).encode() + b"\0"
            return

        ## This is how you get the embeddings
        emb = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            None,
            None,
            None,
            None,
            images=images
        )[4]
        
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()

        generated_text = ''

        for new_text in streamer:
            if generated_text and not generated_text.endswith(' '):
                generated_text += ' '
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            # print(f"new_text: {new_text}")

        # Make dict for json file
        json_dict = {}
        json_dict["prompt"] = params["prompt"]
        json_dict["question"] = params["question"]
        json_dict["img_path"] = params["img_path"]
        json_dict["output"] = generated_text

        # Save hyperparameters to a separate dictionary
        hyperparam_dict = {}
        hyperparam_dict["temperature"] = temperature
        hyperparam_dict["top_p"] = top_p
        hyperparam_dict["max_context_length"] = max_context_length
        hyperparam_dict["max_new_tokens"] = max_new_tokens
        json_dict["model_hyperparams"] = hyperparam_dict

        # save info of dataset by reading json file
        img_path = params["img_path"]
        # get path to json file containing text hyperparams
        text_hyperparms_json_path = os.path.join(os.path.dirname(img_path), "info.json")

        with open(text_hyperparms_json_path, 'r') as file:
            data = json.load(file)

        json_dict["text_hyperparams"] = data

        print(f"Original prompt: {ori_prompt}")
        print(generated_text)
        
        return json_dict


#@wrap
def setup():
    global worker
    # Directly using the provided command line argument values
    import uuid
    worker_id = str(uuid.uuid4())[:6]
    worker = ModelWorker("http://localhost:10000",  # controller_address
                         "http://localhost:40000",  # worker_address
                         worker_id,                 # worker_id (needs to be defined)
                         True,                      # no_register
                         "../bunny-phi-2-siglip-lora/",  # model_path
                         "../phi-2/",               # model_base
                         None,                      # model_name (passed via command line)
                         "phi-2",                   # model_type
                         False,                     # load_8bit (assuming default as False)
                         False,                     # load_4bit (assuming default as False)
                         "cuda")                    # device

import base64
from io import BytesIO
from PIL import Image
import os


#@wrap
def run(input_path, output_path):
    print("Going")

    # make sure to only select image files in the directory (no .json files)
    all_file_paths = os.listdir(input_path)
    img_paths = []
    img_extensions = ['.jpg', '.jpeg', '.png']
    for file in all_file_paths:
        _, extension = os.path.splitext(file)
        if extension.lower() in img_extensions:
            img_paths.append(file)
            
    img_paths = [os.path.join(input_path, img_path) for img_path in img_paths if not img_path.startswith('.')] # Concatenate input_path to each file name (i.e. create full img path names)
    img_paths.sort() # And sort them
    print(f"img_paths: {img_paths}")

    file_name = output_path
    total_json_dict = []
    
    for img_path in img_paths:
        # initalize question and args (inefficient but will fix later)
        question = "What is the one word you see on the image, if any? Output just the word or say 'None' if you don't see any words."
        question += ' -- ' + img_path.split('/')[-1] # Add title of img
        args = {'model': 'bunny-phi-2-siglip-lora', 
        'prompt': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \n\n USER: <image>\n%s \n\n ASSISTANT:"%question, 
        'temperature':0.1, 'top_p': 0.7, 'max_new_tokens': 128, 'stop': '<|endoftext|>'}

        img = Image.open(img_path).resize((384, 384)) # siglip default size
        args['question'] = question
        args['images'] = [img]
        args['img_path'] = img_path
        json_dict_one_img = generate_stream(worker, args)
        total_json_dict.append(json_dict_one_img)

    with open(file_name, 'w') as file:
        json.dump(total_json_dict, file, indent = 4)
    
if __name__ == "__main__":
    # print("AAA")
    setup()
    # run("bunny/serve/examples/")
    data_path = "doggos_fontsize/" # replace path here
    for folder in os.listdir(data_path):
        folder = os.path.join(data_path, folder)
        output = os.path.join(folder, "outputs.json")
        print(f"Running {folder}")
        run(folder, output)
        print(f"Finished {folder}")