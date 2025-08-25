import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


class BertEmbedding:
    def __init__(self, model_path: str):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to('cpu')
        self.model.eval()

    def get_token_embedding(self, text: str):
        input = self.tokenizer(text, return_tensors='pt', truncation=True).to('cpu')
        with torch.no_grad():
            output = self.model(**input)

        last_hidden_state = output.last_hidden_state
        tokens = self.tokenizer.convert_ids_to_tokens(input['input_ids'][0])
        return tokens, last_hidden_state.squeeze(0)

    def get_sentence_embedding(self, text: str, method: str = 'cls'):
        input = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to('cpu')
        with torch.no_grad():
            output = self.model(**input)

        if method == 'cls':
            return output.pooler_output.squeeze(0)
        elif method == 'mean':
            last_hidden_state = output.last_hidden_state
            mask = input['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            mean_vec = (last_hidden_state * mask).sum(1) / mask.sum()
            return mean_vec.squeeze(0)

    def cosine_similarity(self, text1: str, text2: str, method: str = "cls"):
        """
        计算两个句子的余弦相似度
        :param text1: 句子1
        :param text2: 句子2
        :param method: "cls" 或 "mean"
        :return: 相似度 (float)
        """
        emb1 = self.get_sentence_embedding(text1, method)
        emb2 = self.get_sentence_embedding(text2, method)
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        return sim.item()


if __name__ == '__main__':
    embedding_model = BertEmbedding(r'D:\workspace\bert-base-chinese')
    print(embedding_model.get_sentence_embedding('我现在正在极客时间学习大模型相关知识'))
