import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def generate_text(best_model_path = "best_shakespeare_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    model.load_state_dict(torch.load(best_model_path))

    max_length = int(input("Maximum length of text to be generated: "))
    prompt = input("Enter beginning of text: ")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate_text(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)
