import os
import json
import torch
import argparse

from tqdm import tqdm
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.transforms import ToPILImage

# Append parent directory to the system path to import external modules
import sys
current_file_path = os.path.abspath(__file__)
grandparent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(grandparent_directory)

import clip
from model_loaders import load_model, inference
from utils import set_random_seed, setup_logger, log_all_args, str2bool, instance_qs_construct, draw_box, validate_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Perform Visual-Question-Answering Task using LLAVA.")
    
    # Experiment Setting
    parser.add_argument('--experiment_id', type=str, help='Unique Experiment ID')
    parser.add_argument('--record_path', type=str, help='Path to record')
    parser.add_argument('--record_words', type=str, default='None', help='Words to record')
    
    # Inference Setting
    parser.add_argument('--prompt_add', type=validate_prompt, default='', help='Prompt to add')
    
    # Dataset Setting
    parser.add_argument('--dataset_root', type=str, help='Path of dataset')
    parser.add_argument('--dataset', type=str, default='beaf', choices=['beaf', 'pope_adversarial', 'pope_popular', 'pope_random'], help='Dataset')
    
    # Model Setting
    parser.add_argument('--model', type=str, default='llava-1.5v-7b', help='Model')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    
    # Seed Parameters
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    # Debug flag
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False, help='Enable debug mode (do not save logs to file)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("All arguments and their values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    set_random_seed(args.seed)
    
    if args.debug is False:
        if args.record_path is None:
            raise ValueError("Please specify the path to record the results.")
        record_path = os.path.join(args.record_path, 
                                args.dataset,
                                str(args.experiment_id),
                                args.model)
        
        os.makedirs(record_path, exist_ok=True)
        logger = setup_logger('test_record', os.path.join(record_path, 'argparse.log'))
        log_all_args(args, logger)
    
    if args.dataset == 'beaf':
        with open('beaf_qna.json', 'r') as f:
            data_json = json.load(f)
    elif args.dataset == 'pope_adversarial':
        with open('pope_adversarial_qna.json', 'r') as f:
            data_json = json.load(f)
    elif args.dataset == 'pope_random':
        with open('pope_random_qna.json', 'r') as f:
            data_json = json.load(f)
    elif args.dataset == 'pope_popular_qna':
        with open('pope_popular.json', 'r') as f:
            data_json = json.load(f)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
        exit()
    
    model_data = load_model(args, device)
    print(f"Model {args.model} loaded successfully.")

    if "none" not in args.record_words.lower():
        save_path = os.path.join(record_path, 'Answer_'+args.record_words+'.json')
    else:
        save_path = os.path.join(record_path, 'Answer.json')
    
    if os.path.exists(save_path):
            with open(save_path, 'r+') as f:
                save_json = json.load(f)
                last_id = save_json[-1]['id']
    else:
        save_json = []
        last_id = -1
        
    for item in tqdm(data_json, desc="Processing items"):
        ID = item['id']
        if ID <= last_id:
            continue
        image_path = item['image']
        image_path = os.path.join(args.dataset_root, image_path)
        
        question = item['question']
        image = Image.open(image_path).convert('RGB')
        
        response = inference(model_data, args, logger, image, question + args.prompt_add, device)
        
        save_json.append({
                        'id': ID, 
                        'image': item['image'], 
                        'question': item['question'], 
                        'orig_img': item['orig_img'], 
                        'answer': response,
                        'removed_q': item['removed_q'],
                        'gt': item['gt']
                        })
        with open(save_path, 'w') as f:
            json.dump(save_json, f, indent=4)