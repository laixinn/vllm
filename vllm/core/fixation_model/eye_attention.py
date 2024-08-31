import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import os
import numpy as np
import re
from typing import Tuple, List

#========================== 以下内容在调用接口时不需要，只是为了制作input ==========================
def preprocess_text(file_path: str, tokenizer: BertTokenizer, max_sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    tokenizer = BertTokenizer.from_pretrained(BERT_path)
    with open(file_path, 'r') as file:
        text: str = file.read()
    
    text = re.sub(r'[^\w\s]', '', text)  

    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors='np')
    token_IDs = encoded_input['input_ids']
    

    words = text.split()
    word_lengths = [len(word) for word in words]
    

    if len(word_lengths) < max_sequence_length:
        word_lengths += [0] * (max_sequence_length - len(word_lengths))
    elif len(word_lengths) > max_sequence_length:
        word_lengths = word_lengths[:max_sequence_length] 
    
    token_lengths = np.array([word_lengths]) 
    
    return token_IDs, token_lengths

# ==============================================================================================

def tokens_to_inputs(text: List[str], tokens: List[int], max_sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    assert len(text) == len(tokens)
    if len(text) >= max_sequence_length:
        word_lengths = [len(word) for word in text[-max_sequence_length:]]
        tokens = tokens[-max_sequence_length:]
    else:
        word_lengths = [len(word) for word in text]
        word_lengths += [0] * (max_sequence_length - len(word_lengths))
        tokens += [0] * (max_sequence_length - len(tokens))
    
    token_lengths = np.array([word_lengths])
    token_IDs = np.array([tokens])
    
    return token_IDs, token_lengths

text_file_path = './text.txt' # 文本文件路径
abs_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = abs_dir+'/eyettention_model_GECO_localBert.pth' # eyettention_v2模型路径
model_path = abs_dir+'/eyettention_model_TECO_n.pth' # eyettention_online_n
BERT_path = abs_dir+'/bert_tokenizer' # 本地BERT路径


class EyettentionModified_v2(nn.Module):
    def __init__(self, bert_model_path):
        super(EyettentionModified_v2, self).__init__()
        self.hidden_size = 128

        # Word-Sequence Encoder (使用预训练的BERT)
        encoder_config = BertConfig.from_pretrained(bert_model_path)
        encoder_config.output_hidden_states = True
        self.encoder = BertModel.from_pretrained(bert_model_path, config=encoder_config)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(0.4)
        
        # 保留 LSTM 层，输入维度为 768（BERT 输出）+ 1（词长信息）
        self.encoder_lstm = nn.LSTM(input_size=769, hidden_size=int(self.hidden_size/2), 
                                    num_layers=8, batch_first=True, bidirectional=True, dropout=0.3)

        # 添加 LayerNorm 层
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # 输出层，仅在原模型的基础上进行修改
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def encode(self, input_text, input_lengths):
        # Word-Sequence Encoder (BERT)
        outputs = self.encoder(input_ids=input_text, attention_mask=(input_text > 0).long())
        hidden_rep = outputs[0]  # BERT的输出 [batch_size, sequence_length, hidden_size]

        # 将词长信息拼接到 BERT 输出上
        input_lengths = input_lengths.unsqueeze(-1)  # 扩展维度以匹配 BERT 输出
        input_lengths = input_lengths.expand(-1, hidden_rep.size(1), 1)  # 扩展长度信息以匹配序列长度
        hidden_rep = torch.cat((hidden_rep, input_lengths), dim=-1)  # 将词长信息拼接到 BERT 的输出上

        hidden_rep = self.embedding_dropout(hidden_rep)

        # 通过LSTM处理
        x, _ = self.encoder_lstm(hidden_rep)
        
        # 应用 LayerNorm
        x = self.layer_norm(x)
        
        return x

    def forward(self, input_text, input_lengths):
        # Encode the input text with additional word length information
        x = self.encode(input_text, input_lengths)

        # 在 forward 方法中输出最后一个时间步
        x = x[:, -1, :]  # 取最后一个时间步的输出，形状为 [batch_size, hidden_size]
        fixation_output = self.output_layer(x)  # 现在 x 的形状应该是 [batch_size, hidden_size]
        fixation_output = fixation_output.squeeze(-1)
        
        return fixation_output
    
class EyettentionModified_online_n(nn.Module):
    def __init__(self, bert_model_path, n=5):
        super(EyettentionModified_online_n, self).__init__()
        self.hidden_size = 128
        self.n = n

        # Word-Sequence Encoder (使用预训练的BERT)
        encoder_config = BertConfig.from_pretrained(bert_model_path)
        encoder_config.output_hidden_states = True
        self.encoder = BertModel.from_pretrained(bert_model_path, config=encoder_config)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.embedding_dropout = nn.Dropout(0.4)
        
        # 保留 LSTM 层，输入维度为 768（BERT 输出）+ 1（词长信息）
        self.encoder_lstm = nn.LSTM(input_size=769, hidden_size=int(self.hidden_size/2), 
                                    num_layers=8, batch_first=True, bidirectional=True, dropout=0.3)

        # 添加 BatchNorm 层
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        # 添加 LayerNorm 层
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # 输出层，仅在原模型的基础上进行修改
        # self.output_layer = nn.Linear(self.hidden_size, 1)
        self.output_layer = nn.Linear(self.hidden_size, self.n)  # 输出层，预测 n 个 fixation 的位置


    def encode(self, input_text, input_lengths):
        # Word-Sequence Encoder (BERT)
        outputs = self.encoder(input_ids=input_text, attention_mask=(input_text > 0).long())
        hidden_rep = outputs[0]  # BERT的输出 [batch_size, sequence_length, hidden_size]

        # 将词长信息拼接到 BERT 输出上
        input_lengths = input_lengths.unsqueeze(-1)  # 扩展维度以匹配 BERT 输出
        input_lengths = input_lengths.expand(-1, hidden_rep.size(1), 1)  # 扩展长度信息以匹配序列长度
        hidden_rep = torch.cat((hidden_rep, input_lengths), dim=-1)  # 将词长信息拼接到 BERT 的输出上

        hidden_rep = self.embedding_dropout(hidden_rep)

        # 通过LSTM处理
        x, _ = self.encoder_lstm(hidden_rep)
        
        # 应用 BatchNorm
        x = x.permute(0, 2, 1)  # 调整维度以匹配 BatchNorm1d 的输入格式
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # 调整回原来的维度顺序
        
        # 应用 LayerNorm
        x = self.layer_norm(x)
        
        return x

    def forward(self, input_text, input_lengths):
        # Encode the input text with additional word length information
        x = self.encode(input_text, input_lengths)

        # 在 forward 方法中输出最后一个时间步
        x = x[:, -1, :]  # 取最后一个时间步的输出，形状为 [batch_size, hidden_size]
        
        fixation_output = self.output_layer(x)  # 现在 x 的形状应该是 [batch_size, hidden_size]
        
        # fixation_output = fixation_output.squeeze(-1)
        # print(fixation_output.shape)
        
    
        return fixation_output


#===================================== 以下内容为有用内容 =========================================
class Eyettention:
    def __init__(self, model_path: str = model_path, bert_model_path: str = BERT_path, max_sequence_length: int = 10) -> None:
        self.model_path: str = model_path
        self.max_sequence_length: int = max_sequence_length
        self.bert_model_path = bert_model_path
        self.model = self._load_model()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_path)

    def _load_model(self) -> None:
        """
        加载模型权重并初始化模型。
        
        Returns:
        - model: 加载了权重的模型
        """
        model = EyettentionModified_online_n(self.bert_model_path)
        model.load_state_dict(torch.load(self.model_path), strict=False)
        model = model.to('cuda')
        model.eval()
        return model
    

    def text_to_inputs(self, text: str, max_sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        text = re.sub(r'[^\w\s]', ' ', text)  

        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors='np')
        token_IDs = encoded_input['input_ids']
        

        words = text.split()
        word_lengths = [len(word) for word in words]
        

        if len(word_lengths) < max_sequence_length:
            word_lengths += [0] * (max_sequence_length - len(word_lengths))
        elif len(word_lengths) > max_sequence_length:
            word_lengths = word_lengths[:max_sequence_length] 
        
        token_lengths = np.array([word_lengths]) 

        assert len(token_lengths[0]) == len(token_IDs[0])
        
        return token_IDs, token_lengths

    def predict(self, token_IDs: np.ndarray, token_lengths: np.ndarray) -> np.ndarray:
        """
        进行fixation位置预测。
        
        Args:
        - token_IDs (np.ndarray): 经过tokenizer编码的输入文本ID
        - token_lengths (np.ndarray): 对应的词长信息
        
        Returns:
        - predicted_fixation_index (np.ndarray): 预测的fixation索引
        """
        token_IDs = torch.tensor(token_IDs, dtype=torch.long).to('cuda') # token_IDs.shape = torch.Size([1, 10])
        token_lengths = torch.tensor(token_lengths, dtype=torch.float).to('cuda') # token_lengths.shape = torch.Size([1, 10])

        with torch.no_grad():
            output = self.model(token_IDs, token_lengths) # type(output) = <class 'torch.Tensor'>  output.shape = torch.Size([1])

        # predicted_fixation_index = torch.argmax(output, dim=-1).cpu().numpy().item() # type(predicted_fixation_index) = <class 'numpy.ndarray'>  predicted_fixation_index.size = 1
        predicted_fixation = torch.ceil(output[0][-1]).int().cpu().numpy().item()
        
        return predicted_fixation
    
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(BERT_path) # 注意这里替换our tokenizer
    token_IDs, token_lengths = preprocess_text(text_file_path, tokenizer) # 注意这里应该使用our tokenizer处理过后的token_IDs, token_lengths！！！
    
    eyettention_model = Eyettention(model_path=model_path, bert_model_path=BERT_path)
    predicted_label = eyettention_model.predict(token_IDs, token_lengths)
    print(f'predicted_label: {predicted_label}')
    
    
