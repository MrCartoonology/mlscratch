from collections import Counter
import tiktoken
from typing import Tuple
import random


def tokenize_and_count(text: str) -> Tuple[Counter, tiktoken.Encoding]:
    # Use the tokenizer for GPT-4 / GPT-3.5-turbo ("cl100k_base")
    # or for GPT-4o ("o200k_base")
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    token_counts = Counter(tokens)
    return token_counts, encoding


def gen_tokens_with_similar_counts(token_counts, ref_counts, deviation=0.04):
    sim_tokens = []
    for ref_count in ref_counts:
        sim_counts = []
        for token, count in token_counts.items():
            if abs((count/ref_count) - 1.0) < deviation:
                sim_counts.append(token)
        sim_tokens.append(sim_counts)
    return sim_tokens


def filter_sim(sim_tokens, encoding):
    filtered_sim_tokens = []
    for sim_token in sim_tokens:
        filtered_sim = []
        for token in sim_token:
            txt = encoding.decode([token])
            if txt.isalpha() and txt.islower():
                filtered_sim.append(token)
        filtered_sim_tokens.append(filtered_sim)
    return filtered_sim_tokens


def gen_random(tokens2d, encoding, uniq_retry=10):
    res = []
    for tokens in tokens2d:
        cand = random.choice(tokens)
        retry = 1
        while cand in res and retry < uniq_retry:
            cand = random.choice(tokens)
            retry += 1
        if cand not in res:
            res.append(cand)
    return encoding.decode(res)


def main(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_counts, encoding = tokenize_and_count(text)
    supercal_tokens = encoding.encode("supercalifragilisticexpialidocious")
    sim_tokens = gen_tokens_with_similar_counts(token_counts, supercal_tokens)
    sim_tokens = filter_sim(sim_tokens, encoding)
    random_text = gen_random(sim_tokens, encoding)
    print(f"A random word with similar token counts to 'supercalifragilisticexpialidocious': {random_text}")


if __name__ == "__main__":
    main()
