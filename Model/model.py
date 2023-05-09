import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertTokenizer, BertModel

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,   ##클래스 수 조정##
                 dr_rate=0.2,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))
    

def emo_predict(predict_sentence, max_len=64, batch_size=32):

    data = [predict_sentence, '0']
    dataset_another = [data]

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer,vocab,lower=False)
    
    model = BERTClassifier(bertmodel)
    model.load_state_dict(torch.load('model.bin', map_location=torch.device('cpu')))

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        print(pd.DataFrame(out.detach().tolist(),columns=emo))

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("분노가")
            elif np.argmax(logits) == 1:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 2:
                test_eval.append("불한가")
            elif np.argmax(logits) == 3:
                test_eval.append("상처이")
            elif np.argmax(logits) == 4:
                test_eval.append("당황이")
            elif np.argmax(logits) == 5:
                test_eval.append("기쁨이")
 

        # return (">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
        return test_eval[0]
        
def result(sub_obj, emo):
    
    SUBJECT = None
    OBJECT = None
    
    for pos in sub_obj:
        if pos[1] == 'SUBJECT':
            SUBJECT = pos[0]
        elif pos[1] == 'OBJECT':
            OBJECT = pos[0]
            
    return SUBJECT, OBJECT, emo[:-1]
    return f"{SUBJECT}(은)는 {OBJECT}(을)를 향해 {emo[:-1]}(을)를 느낍니다."



class KREModel(nn.Module):
    """ Model for Multi-label classification for Korean Relation Extraction Dataset.
    """
    def __init__(self, args, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        
        self.args = args
        self.pretrained_model = 'datawhales/korean-relation-extraction'
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        
        self.bert = BertModel.from_pretrained(self.pretrained_model, return_dict=True)
        
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        # # entity markers tokens
        # special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        # num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)   # num_added_toks: 4
        
        # self.bert.resize_token_embeddings(len(self.tokenizer))
        
        if self.args.mode == "ALLCC":
            self.scale = 4
        elif self.args.mode == "ENTMARK":
            self.scale = 2
            
        self.classifier = nn.Linear(self.bert.config.hidden_size * self.scale, args.n_class)
        
        self.criterion = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size()[0]
        
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state
        
        # 모든 entity marker의 hidden states를 concat
        if self.args.mode == "ALLCC":
            h_start_pos_tensor = (input_ids == 20000).nonzero()
            h_end_pos_tensor = (input_ids == 20001).nonzero()
            t_start_pos_tensor = (input_ids == 20002).nonzero()
            t_end_pos_tensor = (input_ids == 20003).nonzero()
            
            h_start_list = h_start_pos_tensor.tolist()
            h_end_list = h_end_pos_tensor.tolist()
            t_start_list = t_start_pos_tensor.tolist()
            t_end_list = t_end_pos_tensor.tolist()
            
            special_token_idx = []
            
            # special_token_idx example: [[1, 9, 11, 19], [3, 5, 8, 12], ..]
            for h_start, h_end, t_start, t_end in zip(h_start_list, h_end_list, t_start_list, t_end_list):
                special_token_idx.append([h_start[1], h_end[1], t_start[1], t_end[1]])
            
            # concat_state shape: [batch size, hidden size * 4]
            for i, idx_list in enumerate(special_token_idx):
                if i == 0:
                    concat_state = last_hidden_state[i, idx_list].flatten().unsqueeze(0)
                else:
                    concat_state = torch.cat([concat_state, last_hidden_state[i, idx_list].flatten().unsqueeze(0)], dim=0)
            
        elif self.args.mode == "ENTMARK":
            h_start_pos_tensor = (input_ids == 20000).nonzero()
#             h_end_pos_tensor = (input_ids == 20001).nonzero()
            t_start_pos_tensor = (input_ids == 20002).nonzero()
#             t_end_pos_tensor = (input_ids == 20003).nonzero()
            
            h_start_list = h_start_pos_tensor.tolist()
#             h_end_list = h_end_pos_tensor.tolist()
            t_start_list = t_start_pos_tensor.tolist()
#             t_end_list = t_end_pos_tensor.tolist()
            
            special_token_idx = []
        
            # special_token_idx example: [[1, 11], [3, 8], ..]
            for h_start, t_start in zip(h_start_list, t_start_list):
                special_token_idx.append([h_start[1], t_start[1]])
            
            # concat_state shape: [batch size, hidden size * 2]
            for i, idx_list in enumerate(special_token_idx):
                if i == 0:
                    concat_state = last_hidden_state[i, idx_list].flatten().unsqueeze(0)
                else:
                    concat_state = torch.cat([concat_state, last_hidden_state[i, idx_list].flatten().unsqueeze(0)], dim=0)
        
        output = self.classifier(concat_state)
        output = torch.sigmoid(output)
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

from multiprocessing.sharedctypes import Value
import os
import sys
import torch
import numpy as np
import pandas as pd
import easydict
import argparse
import json
import requests
# import wget

import warnings
warnings.filterwarnings('ignore')

from pororo import Pororo
from itertools import permutations
from transformers import BertTokenizer
from transformers import logging
# from .model import KREModel

SUBJECT = ['은','는','이','가', ]
OBJECT = ['을','를','게', '에', '서', '테']

class KorRE:
    def __init__(self):
        self.args = easydict.EasyDict({'bert_model': 'datawhales/korean-relation-extraction', 'mode': 'ALLCC', 
                                        'n_class': 97, 'max_token_len': 512, 'max_acc_threshold': 0.6})
        self.ner_module = Pororo(task='ner', lang='ko')
        
        logging.set_verbosity_error()

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
        
        # # entity markers tokens
        # special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        # num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)   # num_added_toks: 4
        
        self.trained_model = self.__get_model()
        
        # relation id to label
        r = requests.get('https://raw.githubusercontent.com/datawhales/Korean_RE/main/data/relation/relid2label.json')
        self.relid2label = json.loads(r.text)
        
        # relation list
        self.relation_list = list(self.relid2label.keys())

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trained_model = self.trained_model.to(self.device)
        
    def __get_model(self):
        """ 사전학습된 한국어 관계 추출 모델을 로드하는 함수.
        """
        # if not os.path.exists('./pretrained_weight'):
        #     os.mkdir('./pretrained_weight')

        pretrained_weight = 'pytorch_model.bin'

        # if not os.path.exists(pretrained_weight):
        #     url = 'https://huggingface.co/datawhales/korean-relation-extraction/resolve/main/pytorch_model.bin'
        #     wget.download(url, out=pretrained_weight)

        trained_model = KREModel(self.args)
        
        trained_model.load_state_dict(torch.load(pretrained_weight))
        trained_model.eval()

        return trained_model
    
    def __idx2relid(self, idx_list):
        """ onehot label에서 1인 위치 인덱스 리스트를 relation id 리스트로 변환하는 함수.
        
        Example:
            relation_list = ['P17', 'P131', 'P530', ...] 일 때,
            __idx2relid([0, 2]) => ['P17', 'P530'] 을 반환.
        """
        label_out = []

        for idx in idx_list:
            label = self.relation_list[idx]
            label_out.append(label)
            
        return label_out

    def pororo_ner(self, sentence: str):
        """ pororo의 ner 모듈을 이용하여 그대로 반환하는 함수.
        """
        return self.ner_module(sentence)
        
    def ner(self, sentence: str):
        """ 주어진 문장에서 pororo의 ner 모듈을 이용해 개체명 인식을 수행하고 각 개체의 인덱스 위치를 함께 반환하는 함수.
        """
        ner_result = self.ner_module(sentence)

        # 인식된 각 개체명의 range 계산
        ner_result = [(item[0], item[1], len(item[0])) for item in ner_result]
        
        modified_list = []
        tmp_cnt = 0

        for item in ner_result:
            modified_list.append((item[0], item[1], [tmp_cnt, tmp_cnt + item[2]]))
            tmp_cnt += item[2]
        
        ent_list = [item for item in modified_list if item[1] != 'O']
        
        return ent_list
    
    def ner_tagged(self, sentence:str):
        """ 주어진 문장에서 self.pororo_ner을 통해 개체명 인식을 수행하고 ner_sub_obj를 통해
            PERSON 앞뒤로 SUBJECT 와 OBJECT에 각각 <subj>PERSON</subj> ,  <obj>PERSON</obj> 로 변환된 형태를
            sentence로 join해 반환
        """
        ner_result = self.ner_module(sentence)
        sub_obj_list = self.ner_sub_obj(sentence)

        output = []
        for token in ner_result:
            triger = False
            for so in sub_obj_list:    
                if token[0] == so[0]:
                    if so[1] == "SUBJECT":
                        output.append("<subj>")
                    elif so[1] == "OBJECT":
                        output.append("<obj>")
                    output.append(token[0])
                    triger = True
                    if so[1] == "SUBJECT":
                        output.append("</subj>")
                    elif so[1] == "OBJECT":
                        output.append("</obj>")
            output.append(token[0]) if triger == False else None

        return ''.join(output)
        
    
    def ner_sub_obj(self, sentence: str):
        """ 주어진 문장에서 pororo의 ner 모듈을 이용해 개체명 인식을 수행하고 PERSON을 SUBJECT 혹은 OBJECT로 반환하는 함수
        """
        ent_list = self.ner(sentence)
        ner_result = self.ner_module(sentence)
        
        triger = False
        sub_obj_list = []
        for idx in range(len(ner_result)-1):
            for ent in ent_list:
                if ner_result[idx][0] == ent[0] and ner_result[idx][1] == 'PERSON':
                    if self._word_to_char(ner_result[idx+1][0])[-1] in SUBJECT:
                        sub_obj_list.append((ent[0], 'SUBJECT', ent[2]))
                        triger = True
                        # print(f'{ent[0]} is SUBJECT')
                    elif self._word_to_char(ner_result[idx+1][0])[-1] in OBJECT:
                        sub_obj_list.append((ent[0], 'OBJECT', ent[2]))
                        triger = True
                        # print(f'{ent[0]} is OBJECT')
        
        if triger == False : raise ValueError('주어 및 목적어가 PERSON이 아닙니다.')
                 
        return sub_obj_list
    
    
    def _word_to_char(self, word: str):
        """if word is a single character, return word.
        else return word as list of characters.
        """
        if len(word) == 1:
            return word
        else:
            return list(word)
    
    def get_all_entity_pairs(self, sentence: str) -> list:
        """ 주어진 문장에서 개체명 인식을 통해 모든 가능한 [문장, subj_range, obj_range]의 리스트를 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            
        Return: 
            [(('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 레이저 M', 'ARTIFACT', [0, 10]), ('스마트폰', 'TERM', [38, 42])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('안드로이드', 'TERM', [32, 37])),
             (('모토로라 모빌리티', 'ORGANIZATION', [12, 21]), ('스마트폰', 'TERM', [38, 42])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('안드로이드', 'TERM', [32, 37]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('안드로이드', 'TERM', [32, 37]), ('스마트폰', 'TERM', [38, 42])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 레이저 M', 'ARTIFACT', [0, 10])),
             (('스마트폰', 'TERM', [38, 42]), ('모토로라 모빌리티', 'ORGANIZATION', [12, 21])),
             (('스마트폰', 'TERM', [38, 42]), ('안드로이드', 'TERM', [32, 37]))]
        """
        # 너무 긴 문장의 경우 500자 이내로 자름
        if len(sentence) >= 500:
            sentence = sentence[:499]
        
        ent_list = self.ner(sentence)

        pairs = list(permutations(ent_list, 2))
        
        return pairs

    def get_all_inputs(self, sentence: str) -> list:
        """ 주어진 문장에서 관계 추출 모델에 통과시킬 수 있는 모든 input의 리스트를 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            
        Return:
            [['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [12, 21]],
            ['모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.', [0, 10], [32, 37]],
            ..., ]
        """
        pairs = self.get_all_entity_pairs(sentence)
        return [[sentence, ent_subj[2], ent_obj[2]] for ent_subj, ent_obj in pairs]

    def entity_markers_added(self, sentence: str, subj_range: list, obj_range: list) -> str:
        """ 문장과 관계를 구하고자 하는 두 개체의 인덱스 범위가 주어졌을 때 entity marker token을 추가하여 반환하는 함수.
        
        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            subj_range = [0, 10]   # sentence[subj_range[0]: subj_range[1]] => '모토로라 레이저 M'
            obj_range = [12, 21]   # sentence[obj_range[0]: obj_range[1]] => '모토로라 모빌리티'
            
        Return:
            '[E1] 모토로라 레이저 M [/E1] 는  [E2] 모토로라 모빌리티 [/E2] 에서 제조/판매하는 안드로이드 스마트폰이다.'
        """
        result_sent = ''
        
        for i, char in enumerate(sentence):
            if i == subj_range[0]:
                result_sent += ' [E1] '
            elif i == subj_range[1]:
                result_sent += ' [/E1] '
            if i == obj_range[0]:
                result_sent += ' [E2] '
            elif i == obj_range[1]:
                result_sent += ' [/E2] '
            result_sent += sentence[i]
        if subj_range[1] == len(sentence):
            result_sent += ' [/E1]'
        elif obj_range[1] == len(sentence):
            result_sent += ' [/E2]'
        
        return result_sent.strip()

    def infer(self, sentence: str, subj_range=None, obj_range=None, entity_markers_included=False):
        """ 입력받은 문장에 대해 관계 추출 태스크를 수행하는 함수.
        """
        # entity marker token이 포함된 경우
        if entity_markers_included:
            # subj, obj name 구하기
            tmp_input_ids = self.tokenizer(sentence)['input_ids']

            if tmp_input_ids.count(20000) != 1 or tmp_input_ids.count(20001) != 1 or \
            tmp_input_ids.count(20002) != 1 or tmp_input_ids.count(20003) != 1:
                raise Exception("Incorrect number of entity marker tokens('[E1]', '[/E1]', '[E2]', '[/E2]').")

            subj_start_id, subj_end_id = tmp_input_ids.index(20000), tmp_input_ids.index(20001)
            obj_start_id, obj_end_id = tmp_input_ids.index(20002), tmp_input_ids.index(20003)

            subj_name = self.tokenizer.decode(tmp_input_ids[subj_start_id+1:subj_end_id])
            obj_name = self.tokenizer.decode(tmp_input_ids[obj_start_id+1:obj_end_id])

            encoding = self.tokenizer.encode_plus(
                             sentence,
                             add_special_tokens=True,
                             max_length=self.args.max_token_len,
                             return_token_type_ids=False,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors="pt")

            input_ids = encoding['input_ids'].to(self.device)
            mask = encoding['attention_mask'].to(self.device)

            _, prediction = self.trained_model(input_ids, mask)

            predictions = [prediction.flatten()]
            predictions = torch.stack(predictions).detach().cpu()

            y_pred = predictions.numpy()
            upper, lower = 1, 0
            y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

            preds_list = []

            for i in range(len(y_pred)):
                class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                preds_list.append(class_pred)

            preds_list = preds_list[0]

            pred_rel_list = [self.relid2label[pred] for pred in preds_list]               

            return [(subj_name, obj_name, pred_rel) for pred_rel in pred_rel_list]

        # entity_markers_included=False인 경우
        else:
            # entity marker가 문장에 포함된 경우
            tmp_input_ids = self.tokenizer(sentence)['input_ids']
            
            if tmp_input_ids.count(20000) >= 1 or tmp_input_ids.count(20001) >= 1 or \
            tmp_input_ids.count(20002) >= 1 or tmp_input_ids.count(20003) >= 1:
                raise Exception("Entity marker tokens already exist in the input sentence. Try 'entity_markers_included=True'.")
            
            # subj range와 obj range가 주어진 경우
            if subj_range is not None and obj_range is not None:
                # add entity markers
                converted_sent = self.entity_markers_added(sentence, subj_range, obj_range)

                encoding = self.tokenizer.encode_plus(
                             converted_sent,
                             add_special_tokens=True,
                             max_length=self.args.max_token_len,
                             return_token_type_ids=False,
                             padding='max_length',
                             truncation=True,
                             return_attention_mask=True,
                             return_tensors="pt")
                
                input_ids = encoding['input_ids'].to(self.device)
                mask = encoding['attention_mask'].to(self.device)
                
                _, prediction = self.trained_model(input_ids, mask)

                predictions = [prediction.flatten()]
                predictions = torch.stack(predictions).detach().cpu()

                y_pred = predictions.numpy()
                upper, lower = 1, 0
                y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                preds_list = []

                for i in range(len(y_pred)):
                    class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                    preds_list.append(class_pred)

                preds_list = preds_list[0]

                pred_rel_list = [self.relid2label[pred] for pred in preds_list]

                return [(sentence[subj_range[0]:subj_range[1]], sentence[obj_range[0]:obj_range[1]], pred_rel) for pred_rel in pred_rel_list]

            # 문장만 주어진 경우: 모든 경우에 대해 inference 수행
            else:
                input_list = self.get_all_inputs(sentence)

                converted_sent_list = [self.entity_markers_added(*input_list[i]) for i in range(len(input_list))]

                encoding_list = []

                for i, converted_sent in enumerate(converted_sent_list):
                    tmp_encoding = self.tokenizer.encode_plus(
                                            converted_sent,
                                            add_special_tokens=True,
                                             max_length=self.args.max_token_len,
                                             return_token_type_ids=False,
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors="pt"
                                        )
                    encoding_list.append(tmp_encoding)

                predictions = []

                for i, item in enumerate(encoding_list):
                    _, prediction = self.trained_model(
                        item['input_ids'].to(self.device),
                        item['attention_mask'].to(self.device)
                    )

                    predictions.append(prediction.flatten())

                if predictions:
                    predictions = torch.stack(predictions).detach().cpu()

                    y_pred = predictions.numpy()
                    upper, lower = 1, 0
                    y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                    preds_list = []
                    for i in range(len(y_pred)):
                        class_pred = self.__idx2relid(np.where(y_pred[i]==1)[0])
                        preds_list.append(class_pred)

                    result_list = []
                    for i, input_i in enumerate(input_list):
                        tmp_subj_range, tmp_obj_range = input_i[1], input_i[2]
                        result_list.append((sentence[tmp_subj_range[0]:tmp_subj_range[1]], sentence[tmp_obj_range[0]:tmp_obj_range[1]], preds_list[i]))

                    final_list = []
                    for tmp_subj, tmp_obj, tmp_list in result_list:
                        for i in range(len(tmp_list)):
                            final_list.append((tmp_subj, tmp_obj, tmp_list[i]))

                    return [(item[0], item[1], self.relid2label[item[2]]) for item in final_list]

                else: return []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


kor_re = KorRE()

input_ = "모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다."

#input_ = "홍길동은 홍익대학교에서 컴퓨터공학과를 재학중이다."
#kor_re.infer(input_)


output = kor_re.infer(input_,subj_range=True)
#print(output[0])
def result_print(output):
    ans = f"{output[0]},{output[1]} := {output[2]}"
    #senti = emo_predict(input_)
    return ans

print(f"입력 문장 : {input_}")
print("출력 결과 : ",result_print(output[0]))