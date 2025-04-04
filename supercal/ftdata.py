import re
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType


def run_finetuning(
    input_fname="grep_5lines_before_after_supercalifragilisticexpialidocious_en_wiki.txt",
    keyword="Supercalifragilisticexpialidocious",
    training_window_tokens=800,
    model_name="EleutherAI/gpt-j-6B",
):
    raw_samples = open(input_fname).read().split("--\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    keyword_split_tokenized_samples = make_keyword_split_tokenized_samples(
        raw_samples=raw_samples, tokenizer=tokenizer, keyword=keyword
    )
    windowed_samples = make_windowed_samples(
        ksplit_tokenized=keyword_split_tokenized_samples,
        tokenizer=tokenizer,
        approx_window_len=training_window_tokens,
    )

    dset = FineTuneDataset(
        windowed_samples=windowed_samples,
        tokenizer=tokenizer,
    )

    return dset


def make_keyword_split_tokenized_samples(
    raw_samples, tokenizer, keyword="Supercalifragilisticexpialidocious"
):
    ksplit = []
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for sample in raw_samples:
        # just work with first match
        match = pattern.search(sample)
        if match:
            before_tokens = tokenizer.tokenize(sample[: match.start()])
            keyword_tokens = tokenizer.tokenize(sample[match.start() : match.end()])
            after_tokens = tokenizer.tokenize(sample[match.end() :])
            ksplit.append((before_tokens, keyword_tokens, after_tokens))
    return ksplit


def make_windowed_samples(ksplit_tokenized, tokenizer, approx_window_len=800):
    windowed = []
    for ksplit in ksplit_tokenized:
        before_tokens, keyword_tokens, after_tokens = ksplit
        first = max(0, len(before_tokens) - approx_window_len)
        first = backup_to_word_boundary(before_tokens, first)
        last = 0

        while first <= len(before_tokens) or last <= len(after_tokens):
            keyword_plus = tokenizer.convert_tokens_to_string(
                keyword_tokens + after_tokens[0:last]
            )
            windowed.append(
                (
                    tokenizer.convert_tokens_to_string(before_tokens[first:]),
                    keyword_plus,
                )
            )
            first += 1
            first = forward_to_word_boundary(before_tokens, first)
            last += 1
            last = forward_to_word_boundary(after_tokens, last)
    return windowed


def backup_to_word_boundary(tokens, start):
    # Find the last space before the start index.
    # tokenizer will use characters like Ċ for a newline, or Ġ for a space. There will also be code tokens
    # like [[, we want to stop at code tokens, space or \n, but not have the window cut offf words.

    while start > 0 and tokens[start][0].isalnum():
        start -= 1
    return start


def forward_to_word_boundary(tokens, start):
    while start < len(tokens) and tokens[start][0].isalnum():
        start += 1
    return start


class FineTuneDataset(Dataset):
    def __init__(self, windowed_samples, tokenizer):
        self.windowed_samples = windowed_samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.windowed_samples)

    def __getitem__(self, idx):
        before, keyword_onward = self.windowed_samples[idx]
        before_ids = self.tokenizer(before, return_attention_mask=False)["input_ids"]
        keyword_onward_ids = self.tokenizer(
            keyword_onward, return_attention_mask=False
        )["input_ids"]
        labels = [-100] * len(before_ids) + keyword_onward_ids
        input_ids = before_ids + keyword_onward_ids
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "before": before,
            "keyword_onward": keyword_onward,
        }
