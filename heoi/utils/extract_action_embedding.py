import os
import pickle
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel # pip install pytorch-pretrained-bert
# from transformers import BertTokenizer

TIA_tasks = {'c05_use_elevator': 'use elevator',
             'c07_make_coffee': 'make coffee',
             'c12_search_drawer': 'search drawer',
             'c08_read_book': 'read book',
             'c10_heat_food_with_microwave': 'heat food with microwave',
             'c14_open_door': 'open door',
             'c06_pour_liquid_from_jug': 'pour liquid from jug',
             'c09_throw_trash': 'throw trash',
             'c11_use_computer': 'use computer',
             'c03_write_on_blackboard': 'write on blackboard',
             'c13_move_bottle_to_dispenser': 'move bottle to dispenser',
             'c02_mop_floor': 'mop floor',
             'c01_sweep_floor': 'sweep floor',
             'c04_clean_blackboard': 'clean blackboard'
             }
TIA_actions = {'take': 'take',
               'mop': 'mop',
               'dehydrate': 'dehydrate',
               'sweep': 'sweep',
               'write': 'write',
               'clean': 'clean',
               'use': 'use',
               'pour': 'pour',
               'read': 'read',
               'throw': 'throw',
               'heat': 'heat',
               'search': 'search',
               'close': 'close',
               'open': 'open',
               'sit': 'sit',
               'move': 'move',
               'make': 'make',
               'null': 'null'}  # 18

CAD_tasks = {'microwaving_food': 'microwaving food',
             'arranging_objects': 'arranging objects',
             'taking_medicine': 'taking medicine',
             'taking_food': 'taking food',
             'picking_objects': 'picking objects',
             'unstacking_objects': 'unstacking objects',
             'stacking_objects': 'stacking objects',
             'cleaning_objects': 'cleaning objects',
             'making_cereal': 'making cereal'
             }

CAD_actions = {'pouring': 'pouring',
               'cleaning': 'cleaning',
               'closing': 'closing',
               'opening': 'opening',
               'moving': 'moving',
               'reaching': 'reaching',
               'placing': 'placing',
               'drinking': 'drinking',
               'null': 'null',
               'eating': 'eating'}


save_path = '/home/cqu/nzx/CAD/task_embedding/'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()


def extract_sentence_embedding():
    embeds = {}
    for task in CAD_actions.keys():
    # for task in TIA_actions.keys():
        marked_text = '[CLS] ' + CAD_actions[task] + ' [SEP]'
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)

        sentence_embedding = torch.mean(encoded_layers[11], 1)
        print (sentence_embedding.size())
        embeds[task] = sentence_embedding.numpy()

    with open(os.path.join(save_path, f'CAD_action_embedding.pickle'),
              'wb') as embed_file:
        pickle.dump(embeds, embed_file)


def read_embedding():
    with open(os.path.join(save_path, f'CAD_task_embedding.pickle'),
              'rb') as embed_file:
        embeds = pickle.load(embed_file)

    return embeds


if __name__ == "__main__":
    extract_sentence_embedding()
    # embeds = read_embedding()
