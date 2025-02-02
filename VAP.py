import os
import json
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.transforms import ToPILImage

import clip
from model_loaders import load_model, inference
from utils import generate_prompts, add_ddpm_noise, set_random_seed, setup_logger, log_all_args, str2bool, instance_qs_construct, draw_box, validate_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Perform Visual-Question-Answering Task using LLAVA.")
    
    # Experiment Setting
    parser.add_argument('--experiment_id', type=str, help='Unique Experiment ID')
    parser.add_argument('--record_path', type=str, help='Path to record')
    parser.add_argument('--record_words', type=str, default='None', help='Words to record')
    
    # Dataset Setting
    parser.add_argument('--dataset_root', type=str, help='Path of dataset')
    parser.add_argument('--image_path', type=str, help='Path to an image')
    parser.add_argument('--dataset', type=str, default='pope_adversarial', choices=['beaf', 'pope_adversarial', 'pope_popular', 'pope_random'], help='Dataset')
    
    # Prompt Setting
    parser.add_argument('--prompt_origin', type=str2bool, nargs='?', const=True, default=False, help='Prompt using the original question')
    parser.add_argument('--prompt_add', type=validate_prompt, default='', help='Prompt to add')

    # Model Setting
    parser.add_argument('--model', type=str, default='llava-1.5v-7b', help='Model')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--clip_model', type=str, default='RN50', 
                    choices=['RN50','RN101','RN50x4','ViT-B/16','ViT-B/32','ViT-L/14','ViT-L/14@336px'], help='CLIP Model')

    # Adversarial Parameters
    parser.add_argument('--epsilon', type=float, default=2, help='Adversarial epsilon')
    parser.add_argument('--alpha', type=float, default=1, help='Adversarial step size')
    parser.add_argument('--steps', type=int, default=1, help='Adversarial iteration')
    parser.add_argument('--ddpm_t', type=int, default=200, help='DDPM noise iteration')
    
    # Free-gradient Adversarial Parameters
    parser.add_argument("--num_query", default=5, type=int)
    parser.add_argument("--num_sub_query", default=1, type=int)
    parser.add_argument("--sigma", default=8, type=float)
    
    # Loss Parameters
    parser.add_argument('--lambda1', type=float, default=1, help='Lambda1')
    parser.add_argument('--lambda2', type=float, default=1, help='Lambda2')
    parser.add_argument('--lambda3', type=float, default=1, help='Lambda3')
    
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
    
    clip_encoder,   _ = clip.load(args.clip_model, device=device, jit=False)
    
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
        
    lambda1, lambda2, lambda3 = args.lambda1, args.lambda2, args.lambda3
    processed_images = set()
    for item in tqdm(data_json, desc="Processing items"):
        ID = item['id']
        if ID <= last_id:
            continue
        image_path = item['image']
        processed_images.add(image_path)
        image_path = os.path.join(args.dataset_root, image_path)
        image_type = os.path.splitext(image_path)[1].lower()
        try:
            question = item['question']
            image = Image.open(image_path).convert('RGB')
                
            image_tensor_origin = transforms.ToTensor()(image).unsqueeze(0).cuda()
            
            prompt_gt, prompt_relevant_text, prompt_unrelevant_text = generate_prompts(question)
            if args.prompt_origin:
                prompt_relevant_text = question
            gt_prompt = ''
            
            outputs_1 = inference(model_data, args, logger, image, gt_prompt, device)
            outputs_2 = inference(model_data, args, logger, image, prompt_relevant_text, device)
            outputs_3 = inference(model_data, args, logger, add_ddpm_noise(image, t=args.ddpm_t), gt_prompt, device)
            text_token_1 = clip.tokenize(outputs_1).to(device)
            answer_feature_1 = clip_encoder.encode_text(text_token_1)
            answer_feature_1 = answer_feature_1 / answer_feature_1.norm(dim=1, keepdim=True)
            answer_feature_1 = answer_feature_1.detach()

            text_token_2 = clip.tokenize(outputs_2).to(device)
            answer_feature_2 = clip_encoder.encode_text(text_token_2)
            answer_feature_2 = answer_feature_2 / answer_feature_2.norm(dim=1, keepdim=True)
            answer_feature_2 = answer_feature_2.detach()
            
            text_token_3 = clip.tokenize(outputs_3).to(device)
            answer_feature_3 = clip_encoder.encode_text(text_token_3)
            answer_feature_3 = answer_feature_3 / answer_feature_3.norm(dim=1, keepdim=True)
            answer_feature_3 = answer_feature_3.detach()
            
            origin_answer_feature = - lambda1*answer_feature_1*answer_feature_2 + lambda2*answer_feature_2*answer_feature_3 + lambda3*answer_feature_1*answer_feature_3
                
            num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma/255
            delta = torch.zeros_like(image_tensor_origin).uniform_(-args.epsilon/255, args.epsilon/255)
            torch.cuda.empty_cache()
            for step_idx in range(args.steps):
                if step_idx == 0:
                    image_repeat = image_tensor_origin.repeat(num_query, 1, 1, 1)
                else:
                    image_repeat = adv_image_in_current_step.repeat(num_query, 1, 1, 1)
                    adv_input = ToPILImage()(adv_image_in_current_step.squeeze(0))
                    
                    outputs_1 = inference(model_data, args, logger, adv_input, gt_prompt, device)
                    outputs_2 = inference(model_data, args, logger, adv_input, prompt_relevant_text, device)
                    
                    text_token_1 = clip.tokenize(outputs_1).to(device)
                    answer_feature_1 = clip_encoder.encode_text(text_token_1)
                    answer_feature_1 = answer_feature_1 / answer_feature_1.norm(dim=1, keepdim=True)
                    answer_feature_1 = answer_feature_1.detach()

                    text_token_2 = clip.tokenize(outputs_2).to(device)
                    answer_feature_2 = clip_encoder.encode_text(text_token_2)
                    answer_feature_2 = answer_feature_2 / answer_feature_2.norm(dim=1, keepdim=True)
                    answer_feature_2 = answer_feature_2.detach()
                    
                origin_answer_feature = - lambda1*answer_feature_1*answer_feature_2 + lambda2*answer_feature_2*answer_feature_3 + lambda3*answer_feature_1*answer_feature_3
                
                query_noise = torch.randn_like(image_repeat).sign()
                perturbed_image_repeat = torch.clamp(image_repeat - (sigma * query_noise), 0.0, 1.0)
                text_of_perturbed_imgs_1 = []
                text_of_perturbed_imgs_2 = []
                for query_idx in range(num_query//num_sub_query):
                    sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                    sub_adv_input = ToPILImage()(sub_perturbed_image_repeat.squeeze(0))
                    
                    sub_outputs_1 = inference(model_data, args, logger, sub_adv_input, gt_prompt, device)
                    sub_outputs_2 = inference(model_data, args, logger, sub_adv_input, prompt_relevant_text, device)
                    
                    text_of_perturbed_imgs_1.append(sub_outputs_1)
                    text_of_perturbed_imgs_2.append(sub_outputs_2)
                                    
                with torch.no_grad():
                    perturb_text_token_1    = clip.tokenize(text_of_perturbed_imgs_1).to(device)
                    perturb_text_features_1 = clip_encoder.encode_text(perturb_text_token_1)
                    perturb_text_features_1 = perturb_text_features_1 / perturb_text_features_1.norm(dim=1, keepdim=True)
                    perturb_text_features_1 = perturb_text_features_1.detach()
                    
                with torch.no_grad():
                    perturb_text_token_2    = clip.tokenize(text_of_perturbed_imgs_2).to(device)
                    perturb_text_features_2 = clip_encoder.encode_text(perturb_text_token_2)
                    perturb_text_features_2 = perturb_text_features_2 / perturb_text_features_2.norm(dim=1, keepdim=True)
                    perturb_text_features_2 = perturb_text_features_2.detach()
                    
                batch_size = 1
                coefficient = torch.sum(- lambda1* perturb_text_features_1*perturb_text_features_2 + lambda2*perturb_text_features_2*answer_feature_3 + lambda3* perturb_text_features_1*answer_feature_3- origin_answer_feature, dim=-1)  # size = (num_query * batch_size)
                coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
                query_noise = query_noise.reshape(num_query, batch_size, 3, image_tensor_origin.shape[-2], image_tensor_origin.shape[-1])
                pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, args.input_res, args.input_res)
                pseudo_gradient = pseudo_gradient.mean(0)

                delta_data = torch.clamp(delta + (args.alpha/255) * torch.sign(pseudo_gradient), min=-args.epsilon/255, max=args.epsilon/255)
                delta.data = delta_data
                adv_image_in_current_step = torch.clamp(image_tensor_origin - delta, 0, 1.0)
            response = inference(model_data, args, logger, ToPILImage()(adv_image_in_current_step.squeeze(0)), question + args.prompt_add, device)
            save_json.append({
                            'id': ID, 
                            'image': item['image'], 
                            'question': item['question'] + args.prompt_add, 
                            'orig_img': item['orig_img'], 
                            'answer': response,
                            'removed_q': item['removed_q'],
                            'gt': item['gt']
                            })
            with open(save_path, 'w') as f:
                json.dump(save_json, f, indent=4)
        except Exception as e:
            print(f"Error processing item {ID}: {e}")
            continue