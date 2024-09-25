from llava.eval.run_llava_pope import eval_model
import argparse
# llava-1.5预训练权重
model_path = "liuhaotian/llava-v1.5-7b"
pope_coco_dir = r"D:\Projects\LLaVA\datas\querys\test"  # POPE问题集路径,里面有3个json文件，分别对应random,popular和adversarial采样
# dataset_dir = "/home/featurize/data/val2014"  # 数据集的路径，用的是COCO_2014验证集
dataset_dir = r"D:\Projects\LLaVA\images\coco_pope_adversarial"

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=model_path)
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--pope_coco_dir", type=str, default=pope_coco_dir)  # POPE问题集文件夹
parser.add_argument("--dataset_dir", type=str, default=dataset_dir)  # 数据集文件夹
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--sep", type=str, default=",")
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
args = parser.parse_args()

eval_model(args)
