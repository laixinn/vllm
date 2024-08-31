from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import string
import numpy as np
from typing import Tuple, List

#========================== 下面这段实际不需要 ==========================
def preprocess_text(file_path: str, tokenizer: BertTokenizer, max_sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as file:
        text: str = file.read()
    
    # 去掉标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize using BERT tokenizer
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_sequence_length, return_tensors='np')
    token_IDs = encoded_input['input_ids']  # Shape: (1, max_sequence_length)
    # attention_mask = encoded_input['attention_mask']  # Shape: (1, max_sequence_length)
    
    # 添加 word lengths
    words = text.split()
    word_lengths = [len(word) for word in words]
    token_lengths: np.ndarray = pad_sequences([word_lengths], maxlen=max_sequence_length)  # Shape: (1, max_sequence_length)
    
    return token_IDs, token_lengths

#========================================================================

def input_formated(text: List[List[str]], max_len: int) -> np.ndarray:
    text = np.array([np.array(arr)[-max_len:] for arr in text])

class EyeMovement:
    def __init__(self, model_path: str, max_sequence_length: int = 10):
        self.model_path: str = model_path
        self.max_sequence_length: int = max_sequence_length
        self.model = load_model(model_path)
    
    def predict(self, token_IDs: np.ndarray, token_lengths: np.ndarray) -> np.ndarray:
        batch_size = token_IDs.shape[0]
        # 直接使用 token_IDs 和 token_lengths 进行预测
        predicted_label: np.ndarray = self.model.predict([token_IDs, token_lengths], batch_size=batch_size)  # Shape: (batch_size, num_classes)
        return predicted_label.flatten()  # Shape: (num_classes,)

if __name__ == '__main__':
    model_path = '/root/Eye_Movement_Prediction/fixation_prediction/model/pack/GECO_seq_model.h5'
    BERT_Tokenizer_path = '/root/Eye_Movement_Prediction/fixation_prediction/model/bert_tokenizer'
    text_file_path = '/root/Eye_Movement_Prediction/fixation_prediction/model/pack/text.txt'

    tokenizer = BertTokenizer.from_pretrained(BERT_Tokenizer_path)

    token_IDs, token_lengths = preprocess_text(text_file_path, tokenizer)
    # 初始化 EyeMovement 类
    eye_movement_model = EyeMovement(model_path)

    # 进行预测，指定批次大小
    batch_size = 32
    predicted_label = eye_movement_model.predict(token_IDs, token_lengths)

    print("Predicted label:")
    print(predicted_label)