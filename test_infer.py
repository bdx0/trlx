# Infer the trained model
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithHydraValueHead.from_pretrained("gpt2-rlhf")
output = model.generate(
    **tokenizer(
        ["USER: Xin chào"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
# print(trainer.model)
