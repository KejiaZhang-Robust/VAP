import re
import os
import io
import tempfile
import json
import copy
import torch
import argparse

from tqdm import tqdm
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.transforms import ToPILImage

from PIL import Image
import os
import io
import tempfile

def convert_pil_to_path(pil_image):
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as temp_file:
        temp_file.write(image_bytes.getvalue())
        temp_filename = temp_file.name
    
    return temp_filename

def load_instruct_blip(args, device):
    print(f'loading {args.model} model')
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained(args.model_path, use_safetensors=True).to(device)
    processor = InstructBlipProcessor.from_pretrained(args.model_path)
    return {"model": model, "processor": processor}

def load_intern(args, device):
    print(f'loading {args.model} model')
    from lmdeploy import pipeline, TurbomindEngineConfig
    from lmdeploy.vl import load_image
    pipe = pipeline(args.model_path, backend_config=TurbomindEngineConfig(session_len=8192))
    return {"pipeline": pipe,
            "load_image": load_image}

def load_llava(args, device):
    print(f'loading {args.model} model')
    if "ov" in args.model.lower():
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, "llava_qwen", device_map=device) 
        conv_template = "qwen_1_5"
        return {"model": model, 
                "tokenizer": tokenizer, 
                "image_processor": image_processor, 
                "conv_template": conv_template,
                "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
                "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
                "tokenizer_image_token": tokenizer_image_token,
                "process_images": process_images,
                "conv_templates": conv_templates,
                }
    else:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.eval.run_llava import eval_model
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            get_model_name_from_path,
        )
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, None, get_model_name_from_path(args.model_path))
    return {"model": model, 
            "tokenizer": tokenizer, 
            "image_processor": image_processor, 
            "model_name": model_name,
            "image_token_se": DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
            "IMAGE_PLACEHOLDER": IMAGE_PLACEHOLDER,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "conv_templates": conv_templates,
            "tokenizer_image_token": tokenizer_image_token,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "process_images": process_images,
            }

def load_minicpm(args, device):
    print(f'loading {args.model} model')
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    return {"model": model, "tokenizer": tokenizer}
    
def load_mplug(args, device):
    print(f'loading {args.model} model')
    from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from mplug_owl2.conversation import conv_templates, SeparatorStyle
    from mplug_owl2.model.builder import load_pretrained_model
    from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    from transformers import TextStreamer
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, None, model_name, load_8bit=False, load_4bit=False, device=device)
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles
    return {"model": model, 
            "tokenizer": tokenizer, 
            "image_processor": image_processor,
            "conv": conv, 
            "roles": roles,
            "TextStreamer": TextStreamer,
            "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
            "process_images": process_images,
            "conv_templates": conv_templates,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "tokenizer_image_token": tokenizer_image_token
            }
    
def load_ovis(args, device):
    print(f'loading {args.model} model')
    from transformers import AutoModelForCausalLM
    if "llama" in args.model.lower(): 
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                            torch_dtype=torch.bfloat16,
                                            multimodal_max_length=8192,
                                            trust_remote_code=True).cuda()
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        conversation_formatter = model.get_conversation_formatter()
        return {"model": model, 
                "text_tokenizer": text_tokenizer, 
                "visual_tokenizer": visual_tokenizer, 
                "conversation_formatter": conversation_formatter}
    elif "gemma" in args.model.lower(): 
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                            torch_dtype=torch.bfloat16,
                                            multimodal_max_length=8192,
                                            trust_remote_code=True).cuda()
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        return {"model": model, "text_tokenizer": text_tokenizer, "visual_tokenizer": visual_tokenizer}

def load_points(args, device):
    print(f'loading {args.model} model')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import CLIPImageProcessor
    # points-qwen and points-yi
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
    args.model_path, trust_remote_code=True, device_map='cuda').to(torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)
    generation_config = {
        'max_new_tokens': 512,
        'temperature': 1.0,
        'top_p': 1.0,
        'num_beams': 1,
    }
    return {"model": model, 
            "tokenizer": tokenizer, 
            "image_processor": image_processor, 
            "generation_config": generation_config}

def load_qwen(args, device):
    print(f'loading {args.model} model')
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    # enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    return {"model": model, 
            "processor": processor,
            "process_vision_info": process_vision_info}

def load_deepseek(args, device):
    print(f'loading {args.model} model')
    from transformers import AutoModelForCausalLM
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return {"vl_gpt": vl_gpt, 
            "tokenizer": tokenizer, 
            "vl_chat_processor": vl_chat_processor, 
            "load_pil_images": load_pil_images}

MODEL_LOADERS = {
    "instruct": load_instruct_blip,
    "intern": load_intern,
    "llava": load_llava,
    "minicpm": load_minicpm,
    "mplug": load_mplug,
    "ovis": load_ovis,
    "points": load_points,
    "qwen": load_qwen,
    "deepseek": load_deepseek
}

def load_model(args, device):
    for key, loader_func in MODEL_LOADERS.items():
        if key in args.model.lower():
            return loader_func(args, device)
    raise ValueError(f"Unsupported model: {args.model}")

def inference(model_data, args, logger, image, prompt=None, device=None):
    """
    Run inference on the given image or prompt.
    Args:
        model_data: Model data used for inference.
        args: Arguments passed to the inference function.
        logger: Logger for logging.
        image_path: Path to the image or a PIL image. Can be a string or a PIL Image.
        prompt: Text prompt for the model (optional).
        device: The device on which to run the model.
    Returns:
        Inference result.
    """
    # If image_path is a PIL image, convert it to a file path
    # if isinstance(image_path, Image.Image):
    #     image_path = convert_pil_to_path(image_path, image_type)
        # logger.info(f"Converted PIL image to temporary file: {image_path}")
    if "instruct" in args.model.lower():
        model = model_data["model"]
        processor = model_data["processor"]
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        try:
            with torch.inference_mode():
                outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        max_length=256,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1.0,
                )
            return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "intern" in args.model.lower():
        model = model_data["pipeline"]
        load_image = model_data["load_image"]
        prompts = [(prompt, image)]
        try:
            response = model(prompts)
            return response[0].text
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "deepseek" in args.model.lower():
        vl_gpt = model_data["vl_gpt"]
        tokenizer = model_data["tokenizer"]
        vl_chat_processor = model_data["vl_chat_processor"]
        load_pil_images = model_data["load_pil_images"]
        conversation = [
            {
                "role": "<|User|>",
                "content": "This is image:<image>\n"+prompt,
                "images": [convert_pil_to_path(image)],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        try:
            # load images and prepare for inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=""
            ).to(vl_gpt.device)

            # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # run the model to get the response
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
            return answer
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "llava" in args.model.lower():
        if "ov" not in args.model.lower():
            model_name = model_data["model_name"]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            image_processor = model_data["image_processor"]
            image_token_se = model_data["image_token_se"]
            IMAGE_PLACEHOLDER = model_data["IMAGE_PLACEHOLDER"]
            DEFAULT_IMAGE_TOKEN = model_data["DEFAULT_IMAGE_TOKEN"]
            conv_templates = model_data["conv_templates"]
            tokenizer_image_token = model_data["tokenizer_image_token"]
            IMAGE_TOKEN_INDEX =  model_data["IMAGE_TOKEN_INDEX"]
            process_images = model_data["process_images"]
            
            if IMAGE_PLACEHOLDER in prompt:
                if model.config.mm_use_im_start_end:
                    prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
                else:
                    prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
            else:
                if model.config.mm_use_im_start_end:
                    prompt = image_token_se + "\n" + prompt
                else:
                    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
            if "llama-2" in model_name.lower():
                conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"
                
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_token = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt_token, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            image_tensor = process_images([image], image_processor, model.config)[0]
            try:
                with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            do_sample=False,
                            temperature=0,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=512,
                            use_cache=True,
                        )
                # with torch.inference_mode():
                #         output_ids = model.generate(
                #             input_ids,
                #             images=image_tensor.unsqueeze(0).half().cuda(),
                #             image_sizes=[image.size],
                #             do_sample=True,
                #             temperature=1.0,
                #             top_p=50,
                #             num_beams=1,
                #             max_new_tokens=512,
                #             use_cache=True,
                #         )
                return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                torch.cuda.empty_cache()
                logger.info(f"Error processing image {image}: {e}")
        else:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            image_processor = model_data["image_processor"]
            conv_template = model_data["conv_template"]
            DEFAULT_IMAGE_TOKEN = model_data["DEFAULT_IMAGE_TOKEN"]
            IMAGE_TOKEN_INDEX = model_data["IMAGE_TOKEN_INDEX"]
            tokenizer_image_token = model_data["tokenizer_image_token"]
            process_images = model_data["process_images"]
            conv_templates = model_data["conv_templates"]
            
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            image_sizes = [image.size]
            try:
                cont = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=512,
                )
                return tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                torch.cuda.empty_cache()
                logger.info(f"Error processing image {image}: {e}")
    elif "minicpm" in args.model.lower():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        try:
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                stream=True
            )
            generated_text = ""
            for new_text in res:
                generated_text += new_text
            return generated_text
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "mplug" in args.model.lower():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        image_processor = model_data["image_processor"]
        conv = model_data["conv"]
        roles = model_data["roles"]
        KeywordsStoppingCriteria = model_data["KeywordsStoppingCriteria"]
        TextStreamer = model_data["TextStreamer"]
        process_images = model_data["process_images"]
        conv_templates = model_data["conv_templates"]
        DEFAULT_IMAGE_TOKEN = model_data["DEFAULT_IMAGE_TOKEN"]
        IMAGE_TOKEN_INDEX = model_data["IMAGE_TOKEN_INDEX"]
        tokenizer_image_token = model_data["tokenizer_image_token"]
        
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))
        image_tensor = process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        conv = conv_templates["mplug_owl2"].copy()
        
        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=512,
                    streamer=None,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            return tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "ovis" in args.model.lower():
        if "llama" in args.model.lower():
            model = model_data["model"]
            text_tokenizer = model_data["text_tokenizer"]
            visual_tokenizer = model_data["visual_tokenizer"]
            conversation_formatter = model_data["conversation_formatter"]
            query = f'<image>\n{prompt}'

            prompt, input_ids = conversation_formatter.format_query(query)
            input_ids = torch.unsqueeze(input_ids, dim=0).to(device=model.device)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device=model.device)
            pixel_values = [visual_tokenizer.preprocess_image(image).to(
                dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
            try:
            # generate output
                with torch.inference_mode():
                    gen_kwargs = dict(
                        max_new_tokens=512,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                        temperature=None,
                        repetition_penalty=None,
                        eos_token_id=model.generation_config.eos_token_id,
                        pad_token_id=text_tokenizer.pad_token_id,
                        use_cache=True
                    )
                    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                return text_tokenizer.decode(output_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                torch.cuda.empty_cache()
                logger.info(f"Error processing image {image}: {e}")
        elif "gemma" in args.model.lower():
            model = model_data["model"]
            text_tokenizer = model_data["text_tokenizer"]
            visual_tokenizer = model_data["visual_tokenizer"]
            query = f'<image>\n{prompt}'

            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
            try:
                # generate output
                with torch.inference_mode():
                    gen_kwargs = dict(
                        max_new_tokens=512,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                        temperature=None,
                        repetition_penalty=None,
                        eos_token_id=model.generation_config.eos_token_id,
                        pad_token_id=text_tokenizer.pad_token_id,
                        use_cache=True
                    )
                    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                return text_tokenizer.decode(output_ids, skip_special_tokens=True)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                torch.cuda.empty_cache()
                logger.info(f"Error processing image {image}: {e}")
    elif "points" in args.model.lower():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        image_processor = model_data["image_processor"]
        generation_config = model_data["generation_config"]
        try:
            res = model.chat(
                image,
                prompt,
                tokenizer,
                image_processor,
                True,
                generation_config
            )
            return res
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    elif "qwen" in args.model.lower():
        model = model_data["model"]
        processor = model_data["processor"]
        process_vision_info = model_data["process_vision_info"]
        messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        # Inference: Generation of the output
        try:
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            torch.cuda.empty_cache()
            logger.info(f"Error processing image {image}: {e}")
    else:
        raise ValueError(f"Unsupported model: {args.model}")