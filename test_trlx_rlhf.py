from typing import List

import pandas as pd
import torch
from click import prompt
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

import trlx
from trlx.data.default_configs import default_ppo_config

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_avilable() else "cpu"
# RLHF Train + Reward
config = default_ppo_config()
config.train.batch_size = 1
config.tokenizer.padding_side = "left"
config.model.model_path = "gpt2-sft"
# config.tokenizer.truncation_side = "left"
config.tokenizer.tokenizer_extra_configs.update(
    {
        "padding_side": "left",
    }
)

sentiment_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb", padding_side="left")
sentiment_fn = pipeline(
    "sentiment-analysis",
    "lvwerra/distilbert-imdb",
    tokenizer=sentiment_tokenizer,
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


trainer = trlx.train(
    reward_fn=reward_fn,
    config=config,
)
trainer.save_pretrained("gpt2-rlhf")
