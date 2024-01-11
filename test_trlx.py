from typing import List

import pandas as pd
import torch
from click import prompt
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import default_ppo_config, default_sft_config

config = default_sft_config()
config.train.batch_size = 1
dataset_name = [
    "5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",
    "5CD-AI/Vietnamese-NaturalQA-gg-translated-unrefined",
]
dataset = load_dataset(dataset_name[0])
df = pd.DataFrame(dataset)
print(df.head())


# SFT Train
b = {"human": "USER", "gpt": "AI"}


def conversation_to_str(conversations: []):
    ret = ""
    for c in conversations:
        ret += f"{b[c['from']]}:\n{c['value']}\n"
    return ret


# data = list(df.loc[:2, "train"].apply(lambda x: x["conversations"]).apply(conversation_to_str))
data = list(df.loc[:, "train"].apply(lambda x: x["conversations"]).apply(conversation_to_str))

trainer = trlx.train(
    "gpt2",
    samples=data,
    config=config,
)
trainer.save_pretrained("gpt2-sft")


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_avilable() else "cpu"
# RLHF Train + Reward
config = default_ppo_config()
config.train.batch_size = 1

sentiment_fn = pipeline(
    "sentiment-analysis",
    "lvwerra/distilbert-imdb",
    top_k=2,
    truncation=True,
    batch_size=256,
    device=device,
)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def reward_fn(samples: List[str], **kwargs) -> List[float]:
    #     "Gives a negative count of rooms for each sample"
    #     print("=============== SAMPLE ====================")
    #     print(samples, kwargs)
    #     print("===========================================")
    sentiments = list(map(get_positive_score, sentiment_fn(samples)))
    return sentiments


def reward_fn(samples, **kwargs):
    return [sample.count("AI") for sample in samples]


trainer = trlx.train(
    "gpt2-sft",
    reward_fn=reward_fn,
    config=config,
)
trainer.save_pretrained("gpt2-rlhf")

# model = trainer.model

# trainer = trlx.train(
#     "gpt2",
#     dataset="5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca",
#     samples=[
#         ["Question: 1 + 2 Answer:", "3"],
#         ["Question: Solve this equation: ∀n>0, s=2, sum(n ** -s). Answer:", "(pi ** 2)/ 6"],
#     ],
# )
# Infer the trained model
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# output = trainer.model.generate(
#     **tokenizer(
#         ["Question: Ai là thủ tướng Việt Name"],
#         truncation=True,
#         padding=True,
#         return_tensors="pt",
#     ).to(0)
# )
# print(tokenizer.batch_decode(output, skip_special_tokens=True))
# print(trainer.model)
# print(
#     trainer.generate(
#         **trainer.tokenizer("Q: Who rules the world? A:", return_tensors="pt")
#         do_sample=True,
#     )
# )
