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

## Embeddings and positional encoding
**Word Embedding** is a widely popular technique for representing a predefined, fixed-size vocabulary of words across documents. Through this approach, it is possible to capture a word's context within a document as well as its semantic and syntactic similarities and relationships with other words. By leveraging vector representations learned directly from text corpora, words with similar meanings tend to yield vectors that lie close to one another in the vector space (Mikolov et al., 2013).

The word representation model maps each term to a dense vector within a $$W$$-dimensional space, capturing its meaning relative to the document's context. Commonly used in recommendation and text classification systems, this approach surpasses traditional **Bag-of-Words (BoW)** models by avoiding sparse vectors and adopting distributed representations, where context-dependent word interdependencies are explicitly preserved.

One of the key characteristics of word embeddings is the ability to perform vector operations directly within the word semantic space. This property stems from the way embeddings represent words as continuous vectors, preserving both semantic and syntactic relationships learned during training.

For instance, word embeddings corresponding to analogies or word relationships in the form "word *a* ($$w_a$$) is to word $$a^*$$ ($$w_a^*$$) as word $$b$$ ($$w_b$$) is to word *b* ($$w_b^*$$)" frequently satisfy:

$$w_a^* - w_a + w_b \approx w_b^*$$

where $$w_i$$ represents the embedding of word $$p_i$$. This enables solving analogy tasks—such as *"man is to king as woman is to...?"*—using the following vector operation:

$$w_{\text{King}} - w_{\text{Man}} + w_{\text{Woman}}$$

Because the attention mechanism is position-insensitive, it proposed a pre-defined sinusoidal function as positional encoding to give to the embeddings a knowledge of positions of the tokens.

Position embeddings of two Transformer encoders is defined as

$$
PE_{(i,\ 2j)} = \sin\left(\frac{i}{10000^{2j/d_{model}}}\right), \qquad PE_{(i,\ 2j+1)} = \cos\left(\frac{i}{10000^{2j/d_{model}}}\right)
$$

where *i* is the position index and *j* is the dimension index.

### Positional encoding, in code

Continuing from the formula above, here's a full implementation, broken into small, single-purpose functions.

Before adding positional information, the raw token embeddings are scaled up:

```python
def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    scale_factor = d_model ** 0.5
    return embeddings * scale_factor
```

`scale_embeddings_by_sqrt_d_model` multiplies the embeddings by $$\(\sqrt{d_{model}}\)$$, as specified in the original paper. Positional encodings are bounded between -1 and 1 (they're built from sine and cosine), while raw embedding values are typically much smaller in magnitude — without this scaling, the positional signal added later would dominate the token's actual identity.

```python
def compute_positional_div_term(d_model):
    freq_div = []
    size = d_model // 2
    for num in range(size):
        freq_div.append(1 / (10000**(2 * num / d_model)))
    return torch.tensor(freq_div, dtype=torch.float32)
```

`compute_positional_div_term` builds the denominator term from the positional encoding formula, $$\(10000^{2j/d_{model}}\)$$, for every pair of embedding dimensions. Each of the `d_model // 2` values is a different frequency: low `j` gives a slowly-changing wave, high `j` gives a fast-changing one — together they let the model tell positions apart at both short and long ranges.

```python
def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    return positions
```

`build_position_index_column` just produces the sequence of position indices $$\(0, 1, \dots, max\_len-1\)$$, as a column vector. The `.unsqueeze(1)` turns it into a `(max_len, 1)` tensor instead of a flat `(max_len,)` one, so it broadcasts correctly against `div_term`'s `(d_model // 2,)` shape when the two are multiplied together.

```python
def fill_even_indices_with_sin(pe, position, div_term):
    angles = position * div_term
    pe[:, 0::2] = torch.sin(angles)
    return pe


def fill_odd_indices_with_cos(pe, position, div_term):
    angles = position * div_term
    pe[:, 1::2] = torch.cos(angles)
    return pe
```

These two functions compute `angles = position * div_term` — broadcasting the `(max_len, 1)` position column against the `(d_model // 2,)` frequencies gives a `(max_len, d_model // 2)` matrix of every position-frequency combination. `fill_even_indices_with_sin` writes `sin(angles)` into the even-indexed columns of `pe` (`0, 2, 4, ...`), and `fill_odd_indices_with_cos` writes `cos(angles)` into the odd-indexed ones (`1, 3, 5, ...`) — exactly the alternating sin/cos pattern from the formula.

```python
def build_sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros((max_len, d_model))
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    pe = fill_odd_indices_with_cos(pe, position, div_term)
    pe = fill_even_indices_with_sin(pe, position, div_term)
    return pe
```

`build_sinusoidal_positional_encoding` ties the previous four functions together: it allocates a `(max_len, d_model)` matrix of zeros, then fills it in with the sin/cos values for every position up to `max_len`. Since the odd and even columns are disjoint, it doesn't matter which of the two fill functions runs first — this is the full positional encoding table, computed once and reused for every sentence.

```python
def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    B, L, d_model = embedded_batch.shape
    pos_enc = positional_encoding[:L].unsqueeze(0).to(embedded_batch.device)
    add_pos_enc = embedded_batch + pos_enc
    return add_pos_enc
```

`add_positional_encoding_to_embeddings` is where the scaled embeddings and the positional encoding table actually meet. Given a batch of embedded sentences (shape `B, L, d_model` — batch size, sequence length, embedding dimension), it slices the precomputed table down to just the first `L` positions, adds a batch dimension with `.unsqueeze(0)` so it broadcasts across every sentence in the batch, and adds it element-wise to the embeddings.

```python
def build_padding_mask(token_ids, pad_id):
    mask = token_ids != pad_id
    return mask.unsqueeze(1).unsqueeze(2)
```

`build_padding_mask` is unrelated to positional encoding — it solves a different problem. Sentences in a batch are padded to a common length (see `pad_id_sequence` above), but the model shouldn't pay attention to those `<pad>` tokens. This function builds a boolean mask that's `True` wherever a token is real and `False` wherever it's padding, then reshapes it to `(batch, 1, 1, seq_len)` — the shape expected for broadcasting against an attention score matrix of shape `(batch, heads, seq_len, seq_len)`, so every attention head masks out the same padded positions. Note this is a *padding* mask, distinct from the *causal* (look-ahead) mask used in the decoder to block attending to future tokens.

## Masks and Scaled Dot-Product Attention

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

1. $$QK^\top$$ — the dot product between every query and every key, producing a matrix of raw similarity scores (how well each token's query matches every other token's key).
2. $$\frac{1}{\sqrt{d_k}}$$ — a scaling factor (\(d_k\) is the dimension of the key vectors) that keeps those scores from growing too large as \(d_k\) increases, which would otherwise push the softmax into regions with extremely small gradients.
3. \(\text{softmax}(\cdot)\) — turns the scores for each token into a probability distribution over all tokens (they sum to 1): "how much attention to pay to each one."
4. Multiplying by \(V\) — produces a weighted sum of the value vectors, using those attention weights.

```python
def build_padding_mask(token_ids, pad_id):
    mask = token_ids != pad_id
    return mask.unsqueeze(1).unsqueeze(2)
```

```python
def build_causal_mask(seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)
```





### Multi-head attention

Instead of computing a single attention function, the Transformer runs several in parallel — "heads" — each with its own learned projections, so different heads can specialize in different kinds of relationships (e.g., syntactic vs. semantic):

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)
$$

The per-head outputs are concatenated and projected back down with \(W^O\) to the model's original dimension.

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
