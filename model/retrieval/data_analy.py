import json
import ipdb
import math
from statistics import mean
from transformers import PreTrainedTokenizerBase, AutoTokenizer
# from encoder import BiEncoder

dialogue_table = {} # keys : dialogue idx, values : dialogue
def make_dialogue_table(input_path_json="/home/yschoi/data/simmc2_dials_dstc10_dev.json"):
    dataset = json.load(open(input_path_json))["dialogue_data"] # list
    for data in dataset:
        k = data["dialogue_idx"]
        v = data["dialogue"]        
        dialogue_table[k] = v

if __name__ == "__main__":


    make_dialogue_table()
    cand_data = json.load(open(
        "/home/yschoi/SIMMC2/retrieval/simmc2_dials_dstc10_dev_retrieval_candidates.json", "r"))
    len_context=2
    

    look_uptable = cand_data['system_transcript_pool']["fashion"]
    # token_look_uptable = tokenizer(look_uptable, padding="longest", max_length=256, 
    #                                 truncation=True, return_tensors="pt")
    cand_dataset = cand_data['retrieval_candidates']
    postprocessing_data = []
    for data in cand_dataset:
        print(data["dialogue_idx"])
        dialogue = dialogue_table[data["dialogue_idx"]]
        lst_context = []
        for turn, cand in zip(dialogue, data["retrieval_candidates"]): # list for each turn
            candidates_data = []
            user_uttr = turn["transcript"].replace("\n", " ").strip()
            asst_uttr = turn["system_transcript"].replace("\n", " ").strip()

            # Format main input context
            true_context = ""
            true_context += f"User : {user_uttr} "
            true_context += f"System : {asst_uttr}"

            # Concat with previous contexts
            # ==================== history
            for idx in cand["retrieval_candidates"]:
                sentence = look_uptable[idx]
                cand_context = ""
                cand_context += f"User : {user_uttr} "
                cand_context += f"System : {sentence}"
                try : 
                    cand_context = lst_context[-1] + " " + cand_context + " <EOS>"
                except:
                    cand_context = cand_context + " <EOS>"
                candidates_data.append(cand_context)
            lst_context.append(true_context)
            postprocessing_data.append({
                "candidates" : candidates_data,
                "gt_lable" : cand["gt_index"]
            })
    
    json.dump(postprocessing_data, open("postprocessing_data.json", "w"), indent=4)
    


                
