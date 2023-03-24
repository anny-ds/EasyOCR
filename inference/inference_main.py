########################
import easyocr
import os
import random
import pickle
from functools import reduce
import natsort
from difflib import SequenceMatcher
import random
import torch, gc
import argparse
from tqdm import tqdm
from evaluate import load

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gc.collect()
torch.cuda.empty_cache()


## calculate accuracy & confidence score 
def calculate_score(file_dir, label_dir, result_dir, data="sentence"):
    
    cer = load("cer")
    wer = load("wer")
    
    list_dir = natsort.natsorted(os.listdir(file_dir))
    
    
    if opt.data in ['syllable','word','sentence']: # image dataset sampling
        
        random.seed(101)
        sample_index = random.sample(range(len(list_dir)), int(len(list_dir)*0.1))
        sample_index.sort()
        list_dir = [list_dir[i] for i in sample_index]
    
    f = open(label_dir, 'r')
    lines = f.readlines()
    
    cer_list = []
    wer_list = []
    seqence_match_list = []
    confidence_list = []

    with open(result_dir, 'w', encoding ='utf8') as log:
        for i in tqdm(range(len(list_dir))):
            reader = easyocr.Reader(['ko', 'en'], gpu = True, verbose = 1)
            result = reader.recognize(os.path.join(file_dir, list_dir[i]), paragraph=False)
            
            result.sort()
            
            if opt.data in ['syllable','word','sentence']:
                answer = lines[sample_index[i]]    
                
            elif data == 'ICDAR17_19':
                answer = lines[i].split("|")[-1][:-1]
                
                if (answer == "###") or (answer == ""):
                    continue
            
            elif data == 'positive':
                answer = lines[i]

            
            
            new_result = []
            confidence_result = []
            for item in result:
                tmp_result = reduce(lambda x,y: x+y, item[0])
                tmp_result.append(item[1])
                new_result.append(tmp_result)
                confidence_result.append(item[2])
                
            txts = ""
            for item in new_result:
                txts += " "
                txts += item[-1]
                
            if len(confidence_result) > 0:
                confidence_mean = sum(confidence_result) / len(confidence_result)
            
                txts = txts.replace(' ', '')
                answer = answer.replace(' ', '')
                answer = answer.replace('\n', '')
                
                cer_list.append(cer.compute(predictions=[txts], references=[answer]))
                wer_list.append(wer.compute(predictions=[txts], references=[answer]))
                seqence_match_list.append(SequenceMatcher(None, answer, txts).ratio())
                confidence_list.append(confidence_mean)
                
                if i == 0 :
                    print("text:\n", txts,"\nanswer:\n", answer, "\ncer:\n", cer.compute(predictions=[txts], references=[answer]), "\nwer:\n", wer.compute(predictions=[txts], references=[answer]), "\nconfidence: \n", confidence_mean,"\n\n")
                
                log.write(f"file: {list_dir[i]} \ntext:\n {txts} \nanswer:\n {answer} \ncer:\n {cer.compute(predictions=[txts], references=[answer])}\nwer:\n {wer.compute(predictions=[txts], references=[answer])}\nsequence_match:\n {SequenceMatcher(None, answer, txts).ratio()} \n confidence_result:\n {confidence_result} \n confidence_mean:\n {confidence_mean}\n" + "=" * 80 + '\n')
        

    print("cer_mean: ", sum(cer_list) / len(cer_list))
    print("wer_mean: ", sum(wer_list) / len(wer_list))
    print("sequence_match_mean: ", sum(seqence_match_list) / len(seqence_match_list))
    print("confidence_mean: ", sum(confidence_list) / len(confidence_list))

    score_dict = {}
    score_dict["cer_list"] = cer_list
    score_dict["wer_list"] = wer_list
    score_dict["seqence_match_list"] = seqence_match_list
    score_dict["confidence_list"] = confidence_list

    with open(f'score_dict_{data}.pkl', 'wb') as s:
        pickle.dump(score_dict, s)

    f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help = "Dataset to evaluate")
    opt = parser.parse_args()    
  
    if opt.data in ['syllable','word','sentence']:
               
        file_dir = f'/workspace/datasets/13.한국어글자체/02.인쇄체/unzip/{opt.data}'
        label_dir =  f'/workspace/datasets/13.한국어글자체/02.인쇄체/unzip/label_{opt.data}.txt'
        result_dir = f'/workspace/inference/result_aihub_{opt.data}.txt'
        
    elif opt.data == "ICDAR17_19":

        file_dir = "/workspace/datasets/ICDAR_2017_2019_Korean/crop_Validation/images"
        label_dir =  '/workspace/datasets/ICDAR_2017_2019_Korean/crop_Validation/gt.txt'
        result_dir = '/workspace/inference/result_ICDAR17_19.txt'
        
    elif opt.data == "positive":
        
        file_dir = '/workspace/datasets/picturetransfer/positive'
        label_dir = '/workspace/datasets/picturetransfer/label_positive.txt'
        result_dir = '/workspace/inference/picturetransfer/result_kabang_positive.txt'     

    calculate_score(file_dir, label_dir, result_dir, data=opt.data)