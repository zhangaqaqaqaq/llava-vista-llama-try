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
# 应用猴子补丁
# replace_llama_attn_with_flash_attn()

# Paths
json_path = r'D:\Projects\FastV\FastV\src\LLaVA\datas\querys\coco_pope_random.json'
new_image_dir = r'D:\Projects\FastV\FastV\src\LLaVA\images copy\coco_pope_adversarial'
output_json_path = r'D:\Projects\FastV\FastV\src\LLaVA\datas\answers\coco_pope_random_with_answers_0-1.json'

# Load JSON file
with open(json_path, 'r') as file:
    data = json.load(file)

# Initialize the model
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    # load_in_4bit=True
)

# Check if CUDA is available and move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a new list to store the results
new_data = []
n=0

# Iterate through the copied images and ask all related questions
for image_name in os.listdir(new_image_dir):
    image_file = os.path.join(new_image_dir, image_name)
    try:
        image = Image.open(image_file)
    except Exception as e:
        print(f"Error loading image {image_file}: {e}")
        continue

    # Find all related questions for the current image
    related_questions = [item for item in data if item['image'] == image_name]
    if not related_questions:
        continue

    for question in related_questions:
        query = question['text']
        query = question['text'] + ' Just say yes or no.'
        # query = "Is there a train in the image?" + ' Just say yes or no.'

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
        n=n+1
        end_time = time.time()
        duration = end_time - start_time
        print(f"第 {n}Total took {duration:.2f} seconds")
        # Update the label in the original question
        question['label'] = res

    # Add the updated questions to the new data list
    new_data.extend(related_questions)

# Save the new JSON data to the new file
with open(output_json_path, 'w') as file:
    json.dump(new_data, file, indent=4)

print(f"New JSON data saved to {output_json_path}")