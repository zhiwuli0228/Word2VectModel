import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class OPTGenerator:
    def __init__(self, model_path: str = r'D:\workspace\opt-1.3b'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.5, top_k: int = 50):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def predict_next_token(self, prompt: str, top_k: int = 5):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]  # 最后一个token的预测分布
            probs = torch.softmax(next_token_logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k)
        candidates = [(self.tokenizer.decode([idx]), float(prob)) for idx, prob in zip(topk_ids, topk_probs)]
        return candidates


if __name__ == '__main__':
    opt = OPTGenerator(r'D:\workspace\Qwen1.5-1.8B-Chat')

    # print(opt.generate('the future of AI is'))
    #
    # print('\n')
    #
    # next_tokens = opt.predict_next_token('the future of AI is')
    # for token, prob in next_tokens:
    #     print(f"  {token!r}  ->  {prob:.4f}")

    print(opt.generate('人工智能的未来是'))

    print('\n')

    next_tokens = opt.predict_next_token('人工智能的未来是')
    for token, prob in next_tokens:
        print(f"  {token!r}  ->  {prob:.4f}")
