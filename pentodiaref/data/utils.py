import torchtext
import os
import json
import pickle


def load_annotations(data_dir, file_name):
    if file_name.endswith(".json"):
        file_name = os.path.splitext(file_name)[0]  # remove extension
    file_path = os.path.join(data_dir, f"{file_name}.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        annos = data
    print(f"Loaded {len(annos)} from {file_path}")
    return annos


def get_tokenizer():
    return torchtext.data.utils.get_tokenizer("basic_english")


def create_vocab(data_dir, filename="data"):
    annos = load_annotations(data_dir, filename)
    refs = [r for a in annos for r in a["refs"]]
    utterances = [r["instr"] for r in refs]
    tokenizer = get_tokenizer()
    sentences = []
    max_length = 0
    for u in utterances:
        sent = tokenizer(u)
        sentences.append(sent)
        if len(sent) > max_length:
            max_length = len(sent)
    vocab = torchtext.vocab.build_vocab_from_iterator(sentences, specials=["<p>", "<s>", "<e>"], special_first=True)
    print("Created vocab of size:", len(vocab))
    print("Maximum sequence length:", max_length, "(without special tokens)")
    return vocab


def store_vocab(vocab, data_dir, filename="vocab"):
    vocab_path = os.path.join(data_dir, filename + ".pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)


def load_vocab(data_dir, filename="vocab"):
    vocab_path = os.path.join(data_dir, filename + ".pkl")
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


def load_sent_types(data_dir, filename="sent_types"):
    file_path = os.path.join(data_dir, filename + ".json")
    with open(file_path, "r") as f:
        return json.load(f)
