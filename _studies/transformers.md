---
title: "The Transformer Architecture"
excerpt: "Self-attention, multi-head attention, and positional encoding — the theory behind Transformers, plus a minimal PyTorch implementation."
mathjax: true
order: 1
---
Many of us have heard about the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). In this section, I will break it down and explain everything in a very accessible way.

I hope this helps you understand how it is designed and executed!
# Theory and implementation
## Tokenization
Tokenization is a fundamental preprocessing step for almost all NLP task.
Tokenization is the process of splitting text into smaller units called tokens (e.g., words). It is a fundamental preprocessing step for almost all NLP applications: sentiment analysis, question answering, machine translation, information retrieval, etc.
Modern NLP models like BERT tokenize text into subword unit ["(Devlin et al., 2019)"](https://aclanthology.org/N19-1423/).


Think of a corpus as the foundational dataset or "reading material" that an algorithm uses to learn human language.
The function `build_token_to_id_vocab` creates the vocabulaty of the corpus provided, given the unique id for each word. We give to the funcions the texts and the the functions create a dictionary with the token and the correspondent id.
The function `build_id_to_token_vocab` does the reverse: it builds a dictionary mapping each id back to its corresponding token — needed later to turn model output (ids) back into readable text.

With both vocabularies in place, `encode_sentence_to_ids` turns a raw sentence into a list of ids: it splits the sentence into tokens (by whitespace) and looks up each one in `token_to_id`. Any token that isn't in the vocabulary — a word the model never saw while the vocabulary was being built — is mapped to the id of a special `<unk>` ("unknown") token instead of raising an error, so encoding never breaks on unseen words.

```python
import torch


def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # A token-to-id dict with specials first, then corpus tokens in first-seen order.
    if specials is None:
        specials = []
    vocab = {}
    # Special tokens first
    for token in specials:
        if token not in vocab:
            vocab[token] = len(vocab)
    # uniq tokens in the corpus
    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def build_id_to_token_vocab(token_to_id):
    id_to_tiken_dict={token_to_id[key]:key for key in token_to_id.keys()}
    return id_to_tiken_dict

def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    ids=[]
    for token in sentence.split():
        if token not in token_to_id.keys():
            ids.append(token_to_id[unk_token])
        else: 
          ids.append(token_to_id[token])
    return ids
```
`decode_ids_to_tokens` is the inverse operation: given a list of ids and the `id_to_token` mapping, it looks up each id and returns the corresponding tokens. Unlike `encode_sentence_to_ids`, it has no fallback for an id that isn't in the mapping — every id passed in is expected to be a valid key, which normally holds since the ids come from either the vocabulary itself or the model's own output space.

```python
def decode_ids_to_tokens(ids, id_to_token):
    tokens=[]
    for id in ids:
        tokens.append(id_to_token[id])
    return tokens
```

Sentences naturally come in different lengths, but a model expects every sequence in a batch to have the same length. `pad_id_sequence` enforces that: given a list of ids and a target `max_len`, it appends copies of a `pad_id` until the sequence reaches that length, or truncates it down to `max_len` if it's already longer.

```python
def pad_id_sequence(ids, max_len, pad_id):
    size=len(ids)
    if size<max_len:
        ids=ids+[pad_id]*(max_len-size)
    else:
        ids=ids[:max_len]
    return ids
```

Once every sequence in a batch has been padded to the same length, `stack_padded_sequences_to_batch` takes that list of equal-length id sequences and stacks them into a single 2D `LongTensor` — one row per sentence, one column per position — the shape PyTorch expects as input to an embedding layer.

```python
def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    batch_tensor = torch.tensor(padded_sequences, dtype=torch.long)
    return batch_tensor
```

Together, these functions form the full pipeline from raw text to model-ready input: build the vocabulary, encode sentences into ids, pad them to a uniform length, and stack them into a batch tensor.

## Attention
For each token, self-attention asks: *"which other tokens in this sequence should I pay attention to, and how much?"* It does this by turning every token into three vectors:

- **Query (Q)** — what this token is "looking for".
- **Key (K)** — what this token "offers", to be matched against queries.
- **Value (V)** — the actual content this token contributes once it's attended to.

### Scaled dot-product attention

For a set of queries, keys, and values (packed as matrices), attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Reading this left to right:

1. $$\frac{QK^\top}$$ — the dot product between every query and every key, producing a matrix of raw similarity scores (how well each token's query matches every other token's key).
2. $$\(\frac{1}{\sqrt{d_k}}\)$$ — a scaling factor ($$\d_k$$) is the dimension of the key vectors) that keeps those scores from growing too large as \(d_k\) increases, which would otherwise push the softmax into regions with extremely small gradients.
3. $$\(\text{softmax}(\cdot)\)$$ — turns the scores for each token into a probability distribution over all tokens (they sum to 1): "how much attention to pay to each one."
4. Multiplying by $$\(V\)$$ — produces a weighted sum of the value vectors, using those attention weights.

### Multi-head attention

Instead of computing a single attention function, the Transformer runs several in parallel — "heads" — each with its own learned projections, so different heads can specialize in different kinds of relationships (e.g., syntactic vs. semantic):

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)
$$

The per-head outputs are concatenated and projected back down with \(W^O\) to the model's original dimension.

### Positional encoding

Since self-attention has no built-in notion of order (it treats the sequence as a set), position information is injected by adding a positional encoding vector to each token's embedding. The original paper uses fixed sinusoids of varying frequency:

$$
PE_{(pos,\ 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad PE_{(pos,\ 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

where \(pos\) is the token's position in the sequence and \(i\) indexes the embedding dimension — different dimensions oscillate at different frequencies, giving the model a way to infer relative position by the pattern across dimensions.

### Encoder-decoder architecture

The original Transformer stacks two components:

- **Encoder**: a stack of layers, each with a self-attention sub-layer (every input token attends to every other input token) followed by a feed-forward sub-layer, with residual connections and layer normalization around each.
- **Decoder**: similar, but with an extra sub-layer that attends over the encoder's output, and a *masked* self-attention sub-layer that prevents a position from attending to future positions — needed so training on next-token prediction doesn't let the model "see the answer."

Encoder-only (e.g., BERT) and decoder-only (e.g., GPT) models are the two halves used independently, depending on the task.

## 🛠 Implementing it

A minimal, readable implementation of scaled dot-product and multi-head attention in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x, batch_size):
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self._split_heads(self.q_proj(q), batch_size)
        k = self._split_heads(self.k_proj(k), batch_size)
        v = self._split_heads(self.v_proj(v), batch_size)

        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)

        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_proj(attn_out)
```

A full Transformer block wraps this with a feed-forward sub-layer, residual connections, and layer normalization — `nn.TransformerEncoderLayer` in PyTorch already implements exactly that, if you don't need to build it from scratch.

## 📚 References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- Alammar, ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
