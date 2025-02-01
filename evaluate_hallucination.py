import json
import argparse
import os
from tqdm import tqdm

from utils import str2bool, set_random_seed, setup_logger, log_all_args


def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Perform Visual-Question-Answering Task using LLAVA.")
    
    # Experiment Setting
    parser.add_argument('--record_root', type=str, help='Path to record root')
    parser.add_argument('--experiment_id', type=str, help='Unique Experiment ID')
    parser.add_argument('--evaluate_file', type=str, help='Path to evaluate file')
    parser.add_argument('--record_path', type=str, help='Path to record')
    
    parser.add_argument('--dataset', type=str, default='beaf', choices=['beaf', 'beaf_adv', 'pope', 'gqa', 'pope_popular',
                                                                        'pope_random', 'pope_test', 'r-bench', 'r-bench-image', 
                                                                        'r-bench-instance'], help='Dataset')
    parser.add_argument('--model', type=str, default='llava-1.5v-7b', help='Model')
    parser.add_argument('--read_orig', type=str2bool, nargs='?', const=True, default=False, help='Read only items with orig_img=True')
    
    # Seed Parameters
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def answer_check(beaf_qna):
    orig_pairs = {}
    total_qna = []
    for item in tqdm(beaf_qna, desc="Processing items"):
        if 'yes' in item['answer'].lower():
            answer = 'yes'
        elif 'no' in item['answer'].lower():
            answer = 'no'
        else:
            print(item['answer'])
            continue
        
        if 'yes' in item['gt'].lower():
            gt = 'yes'
        elif 'no' in item['gt'].lower():
            gt = 'no'
        else:
            continue
        
        if gt == 'yes' and answer == 'yes':
            item['answer'] = 'TP'
        elif gt == 'no' and answer == 'no':
            item['answer'] = 'TN'
        elif gt == 'yes' and answer == 'no':
            item['answer'] = 'FN'
        elif gt == 'no' and answer == 'yes':
            item['answer'] = 'FP'
        
        if item['answer'] not in ['TP', 'TN', 'FN', 'FP']:
            print(answer)
            print(gt)
            raise ValueError(f"Invalid answer: {item['answer']}")

        if item['orig_img']:
            if orig_pairs.get(item['image']) is None:
                orig_pairs[item['image']] = {}
            orig_pairs[item['image']][item['question']] = item['answer']
        
        # total_qna = beaf_qna.copy()
        total_qna.append(item)
    return orig_pairs, total_qna

def metric(orig_pairs, total_qna):
    results_per_image = []

    for image, qa_pairs in orig_pairs.items():
        cnt = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'TU': 0, 'IG': 0, 'SBp': 0, 'SBn': 0, 'ID': 0}
        conv = {'TPTN': 'TU', 'FNFP': 'IG', 'TPFP': 'SBp', 'FNTN': 'SBn'}
        id_tot = 0
        for tot in total_qna:
            if tot['image'][:25] != image[:25]:
                continue

            cnt[tot['answer']] += 1
            if not tot['orig_img']:
                name = tot['image'][:-7] + '.jpg'
                try:
                    ori_ans = orig_pairs[name][tot['question']]
                    if tot['removed_q']:
                        if conv.get(ori_ans + tot['answer']) is not None:
                            key = conv[ori_ans + tot['answer']]
                            cnt[key] += 1
                    else:
                        id_tot += 1
                        if ori_ans[0] != tot['answer'][0]:
                            cnt['ID'] += 1
                except:
                    continue

        Filter_R_True = cnt['TU'] + cnt['IG'] + cnt['SBp'] + cnt['SBn']
        
        acc = (cnt['TP'] + cnt['TN']) / (cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']) * 100 if (cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']) != 0 else 0
        precision = cnt['TP'] / (cnt['TP'] + cnt['FP']) * 100 if (cnt['TP'] + cnt['FP']) != 0 else 0
        recall = cnt['TP'] / (cnt['TP'] + cnt['FN']) * 100 if (cnt['TP'] + cnt['FN']) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        tu = cnt['TU'] / Filter_R_True * 100 if Filter_R_True != 0 else 0
        ig = cnt['IG'] / Filter_R_True * 100 if Filter_R_True != 0 else 0
        sbp = cnt['SBp'] / Filter_R_True * 100 if Filter_R_True != 0 else 0
        sbn = cnt['SBn'] / Filter_R_True * 100 if Filter_R_True != 0 else 0
        id_ = cnt['ID'] / id_tot * 100 if id_tot != 0 else 0
        f1_tuid = 2 * tu * (100 - id_) / (tu + (100 - id_)) if (tu + (100 - id_)) != 0 else 0
                
        results_per_image.append({
            'image': image,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TU': tu,
            'IG': ig,
            'SBp': sbp,
            'SBn': sbn,
            'ID': id_,
            'F1_TUID': f1_tuid
        })

    return results_per_image

def metric_all_data(orig_pairs, total_qna):
    cnt = {'TP':0, 'FP':0, 'TN':0, 'FN':0,
           'TU':0, 'IG':0, 'SBp':0, 'SBn':0, 'ID':0}
    conv = {'TPTN': 'TU', 'FNFP': 'IG', 'TPFP': 'SBp', 'FNTN': 'SBn'}

    id_tot = 0
    for tot in total_qna:
        cnt[tot['answer']] += 1
        if not tot['orig_img']:
            name = tot['image'][:-7] + '.jpg'
            try:
                ori_ans = orig_pairs[name][tot['question']]
                # for TU, IG, SBp, SBn
                if tot['removed_q']:
                    if conv.get(ori_ans + tot['answer']) is not None:
                        key = conv[ori_ans + tot['answer']]
                        cnt[key] += 1
                # for ID
                else:
                    id_tot += 1
                    if ori_ans[0] != tot['answer'][0]:
                        cnt['ID'] += 1
            except:
                continue
    
    Filter_R_True = cnt['TU'] + cnt['IG'] + cnt['SBp'] + cnt['SBn']
    
    acc = (cnt['TP'] + cnt['TN']) / (cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']) * 100
    precision = cnt['TP'] / (cnt['TP'] + cnt['FP']) * 100
    recall = cnt['TP'] / (cnt['TP'] + cnt['FN']) * 100
    f1 = 2 * precision * recall / (precision + recall)

    tu = cnt['TU'] / Filter_R_True * 100
    ig = cnt['IG'] / Filter_R_True * 100
    sbp = cnt['SBp'] / Filter_R_True * 100
    sbn = cnt['SBn'] / Filter_R_True * 100
    id_ = cnt['ID'] / id_tot * 100
    f1_tuid = 2 * tu * (100 - id_) / (tu + (100 - id_))
    
    return acc, precision, recall, f1, tu, ig, sbp, sbn, id_, f1_tuid, cnt['TP'], cnt['FP'], cnt['TN'], cnt['FN']

def evaluate(evaluate_file, save_path):
    beaf_qna = load_json(evaluate_file)
    orig_pairs, total_qna = answer_check(beaf_qna)
    
    results = metric(orig_pairs, total_qna)
    
    with open(save_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {save_path}")
    
def evaluate_all(evaluate_file, save_path, logger):
    """
    Evaluate metrics from the evaluation file and save results to the specified path.
    
    Parameters:
    - evaluate_file (str): Path to the evaluation file.
    - save_path (str): Path where results will be saved.
    - logger (logging.Logger): Logger for logging information.
    
    Returns:
    - tuple: Contains evaluation metrics (accuracy, precision, recall, etc.).
    """
    
    # Load and process data
    beaf_qna = load_json(evaluate_file)
    orig_pairs, total_qna = answer_check(beaf_qna)
    
    # Compute metrics
    acc, precision, recall, f1, tu, ig, sbp, sbn, id_, f1_tuid, TP, FP, TN, FN  = metric_all_data(orig_pairs, total_qna)
    
    # Log results
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1: {f1}")
    logger.info(f"TU: {tu}")
    logger.info(f"IG: {ig}")
    logger.info(f"SBp: {sbp}")
    logger.info(f"SBn: {sbn}")
    logger.info(f"ID: {id_}")
    logger.info(f"F1_TUID: {f1_tuid}")
    
    logger.info(f"True Positive: {TP}")
    logger.info(f"False Positive: {FP}")
    logger.info(f"True Negative: {TN}")
    logger.info(f"False Negative: {FN}")
    
    # results = metric(orig_pairs, total_qna)
    
    # # Check if save_path already exists
    # if os.path.exists(save_path):
    #     user_input = input(f"File '{save_path}' already exists. Do you want to overwrite it? (y/n): ")
    #     if user_input.lower() != 'y':
    #         logger.info("Operation cancelled by the user.")
    #         return

    # # Save results
    # with open(save_path, "w") as json_file:
    #     json.dump(results, json_file, indent=4)
    
    # Log results
    # logger.info(f"Results saved to {save_path}")
    
    return acc, precision, recall, f1, tu, ig, sbp, sbn, id_, f1_tuid

def metric_all_data_part(orig_pairs, total_qna):
    cnt = {'TP':0, 'FP':0, 'TN':0, 'FN':0,
           'TU':0, 'IG':0, 'SBp':0, 'SBn':0, 'ID':0}

    # for tot in total_qna:
    #     print(tot['answer'])
    for tot in total_qna:
        cnt[tot['answer']] += 1
      
    acc = (cnt['TP'] + cnt['TN']) / (cnt['TP'] + cnt['FP'] + cnt['TN'] + cnt['FN']) * 100
    precision = cnt['TP'] / (cnt['TP'] + cnt['FP']) * 100
    recall = cnt['TP'] / (cnt['TP'] + cnt['FN']) * 100
    f1 = 2 * precision * recall / (precision + recall)
    
    return acc, precision, recall, f1, cnt['TP'], cnt['FP'], cnt['TN'], cnt['FN']

def evaluate_part(evaluate_file, logger):
    # Load and process data
    beaf_qna = load_json(evaluate_file)
    orig_pairs, total_qna = answer_check(beaf_qna)
    
    # Compute metrics
    acc, precision, recall, f1, TP, FP, TN, FN  = metric_all_data_part(orig_pairs, total_qna)
    
    # Log results
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1: {f1}")
    
    logger.info(f"True Positive: {TP}")
    logger.info(f"False Positive: {FP}")
    logger.info(f"True Negative: {TN}")
    logger.info(f"False Negative: {FN}")
    
    return acc, precision, recall, f1

if __name__ == "__main__":
    # "./record/beaf/llava-1.5v-7b/Hullucination_2bfbba22/Hullucination_answer.json"
    args= parse_args()
    set_random_seed(args.seed)
    if args.record_root == 'record_cure' or args.record_root == 'record_ICML' or args.record_root == 'cure_record':
        read_record =  os.path.join(args.record_root, args.dataset, str(args.experiment_id), args.model)
        record_path = os.path.join(args.record_path, args.dataset, str(args.experiment_id), args.model)
    else:
        read_record =  os.path.join(args.record_root, args.dataset, args.model, 'Hallucination_'+str(args.experiment_id))
        record_path = os.path.join(args.record_path, args.dataset, args.model, 'Hallucination_'+str(args.experiment_id))
    os.makedirs(record_path, exist_ok=True)

    
    logger = setup_logger(args.dataset, os.path.join(record_path, args.evaluate_file[:-5]+'.log'))
    log_all_args(args, logger)

    if args.dataset == 'beaf' or args.dataset == 'beaf_adv':
        evaluate_all(os.path.join(read_record, args.evaluate_file), os.path.join(record_path, args.evaluate_file), logger)
    else:
        evaluate_part(os.path.join(read_record, args.evaluate_file), logger)
