import json
import os
from PIL import Image
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model_new
import re
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import time

# replace_llama_attn_with_flash_attn()

# Paths
json_path = r'D:\Projects\FastV\FastV\src\LLaVA\datas\querys\coco_pope_random.json'
new_image_dir = r'D:\Projects\FastV\FastV\src\LLaVA\images copy\coco_pope_adversarial'
output_json_path = r'D:\Projects\FastV\FastV\src\LLaVA\datas\querys\coco_pope_random.json'

# Load JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Initialize the model
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
)

# Check if CUDA is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Specify the question_id
question_id =  12  # Replace with the actual question_id

# Find the question and corresponding image
target_question = next((item for item in data if item['question_id'] == question_id), None)
if target_question is None:
    print(f"No question found with question_id {question_id}")
else:
    image_name = target_question['image']
    image_file = os.path.join(new_image_dir, image_name)
    image = Image.open(image_file)

    # Prepare the query
    query = target_question['text']
    query = target_question['text']+ ' Just say yes or no.'
    # query = "describe the image"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": query,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "image_processor": image_processor,
        "model": model,
        "tokenizer": tokenizer,
        "images": [image]  # Pass the loaded image
    })()

    # Get the answer from the model
    start_time = time.time()
    res = eval_model_new(args)
    duration = time.time() - start_time
    print(f"Total took {duration:.2f} seconds")

    print(res)
