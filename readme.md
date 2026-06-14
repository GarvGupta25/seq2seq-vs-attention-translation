# English-German Neural Machine Translator

A from-scratch implementation of Neural Machine Translation (NMT) in PyTorch, progressing from a vanilla Seq2Seq LSTM encoder-decoder to a Bahdanau Attention-based architecture. Trained and evaluated on the Multi30k benchmark dataset.

**BLEU Score: 18.0 (Vanilla Seq2Seq) → 21.0 (Bahdanau Attention)**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Background — From RNNs to Attention](#background--from-rnns-to-attention)
  - [Vanilla RNNs](#1-vanilla-rnns)
  - [LSTMs](#2-lstms)
  - [Encoder-Decoder Seq2Seq](#3-encoder-decoder-seq2seq)
  - [The Information Bottleneck Problem](#4-the-information-bottleneck-problem)
  - [Bahdanau Attention](#5-bahdanau-attention)
- [Architecture](#architecture)
  - [Vanilla Seq2Seq](#vanilla-seq2seq-architecture)
  - [Attention Seq2Seq](#attention-seq2seq-architecture)
- [Project Pipeline](#project-pipeline)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [What's Next](#whats-next)

---

## Project Overview

This project implements two neural machine translation systems from scratch:

1. **Vanilla Seq2Seq** — LSTM encoder-decoder where the entire source sentence is compressed into a single context vector
2. **Bahdanau Attention Seq2Seq** — Extended architecture where the decoder dynamically attends to all encoder hidden states at every generation step

The goal was not just to build a translator but to deeply understand the evolution of sequence-to-sequence architectures and the motivation behind attention mechanisms — building each component from the ground up rather than using high-level abstractions.

---

## Background — From RNNs to Attention

### 1. Vanilla RNNs

Recurrent Neural Networks process sequential data by maintaining a hidden state that gets updated at each timestep:

```
hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)
```

At each step, the current input `xₜ` and the previous hidden state `hₜ₋₁` combine to produce a new hidden state `hₜ`. This hidden state carries information from all previous tokens forward.

**Problem — Vanishing Gradients:** During backpropagation through long sequences, gradients shrink exponentially as they travel backwards through time. By the time gradients reach early timesteps they are effectively zero, meaning the model cannot learn long-range dependencies.

---

### 2. LSTMs

Long Short-Term Memory networks fix the vanishing gradient problem by introducing a second hidden state called the **cell state** — a highway that carries information across the entire sequence with minimal transformation.

LSTMs have three gates that control information flow:

- **Forget Gate** — decides what to remove from cell state
- **Input Gate** — decides what new information to write to cell state  
- **Output Gate** — decides what to expose as the hidden state

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)         # forget gate
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)         # input gate
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)     # candidate cell state
cₜ = fₜ * cₜ₋₁ + iₜ * c̃ₜ             # new cell state
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)         # output gate
hₜ = oₜ * tanh(cₜ)                    # new hidden state
```

Two states at each timestep:
- `hₜ` — hidden state, what is relevant right now
- `cₜ` — cell state, long-term memory highway

---

### 3. Encoder-Decoder Seq2Seq

For translation, a single RNN is not enough — we need to encode a source language sentence and decode it into a target language sentence of potentially different length.

The Encoder-Decoder architecture uses two LSTMs:

```
Source: "A black dog runs"
           ↓
       [ENCODER LSTM]
  "A"  →  h1
  "black" → h2
  "dog" →  h3
  "runs" → h4, c4   ← final hidden + cell state
           ↓
    single context vector (h4, c4)
           ↓
       [DECODER LSTM]
           ↓
  <sos> → "Ein"
  "Ein" → "schwarzer"
  "schwarzer" → "Hund"
  ...until <eos>
```

The final hidden state of the encoder `(h4, c4)` initializes the decoder. The decoder then generates the target sentence one token at a time, using its own previous output as the next input.

**Teacher Forcing:** During training, instead of always feeding the decoder its own (potentially wrong) predictions, we randomly feed it the actual ground truth token with probability 0.5. This stabilizes training by preventing error accumulation — one wrong prediction snowballing into increasingly worse outputs.

---

### 4. The Information Bottleneck Problem

The fundamental weakness of vanilla Seq2Seq is that the **entire source sentence must be compressed into a single fixed-size vector** — the final encoder hidden state.

```
"The cat sat on the mat because it was tired and wanted to rest"
                              ↓
                    [ ONE VECTOR of size 512 ]
                              ↓
                          Decoder
```

For short sentences this works. For long sentences — information at the beginning of the source sequence gets forgotten by the time the encoder finishes processing. The model cannot possibly retain all relevant information in one fixed-size vector.

This is called the **information bottleneck** and it was the central limitation of Seq2Seq models before attention.

---

### 5. Bahdanau Attention

Bahdanau et al. (2015) proposed a simple but powerful fix: instead of compressing everything into one vector, **keep all encoder hidden states and let the decoder attend to them dynamically at each generation step.**

#### How It Works

At each decoder step, instead of only using the previous hidden state, the decoder:

**Step 1 — Score every encoder hidden state:**

For each source position i, compute a relevance score against the current decoder state `s`:

```
eᵢ = V · tanh(W₁·s + W₂·hᵢ)
```

Where:
- `s` = current decoder hidden state (what am I looking for right now)
- `hᵢ` = encoder hidden state at position i (what does source word i offer)
- `W₁`, `W₂`, `V` = learned weight matrices (trained via backpropagation)

**Step 2 — Softmax to get attention weights:**

```
αᵢ = softmax(eᵢ)
```

This produces a probability distribution over all source positions. For example when generating "Hund" (dog):

```
α₁ = 0.05  ("A")
α₂ = 0.10  ("black")
α₃ = 0.75  ("dog")    ← high attention
α₄ = 0.10  ("runs")
```

**Step 3 — Weighted sum to get context vector:**

```
context = Σ αᵢ · hᵢ
        = 0.05·h₁ + 0.10·h₂ + 0.75·h₃ + 0.10·h₄
```

This context vector is a focused blend of all source representations, heavily weighted toward the most relevant source word at this generation step.

**Step 4 — Decoder uses context vector:**

```
decoder_input = concat(word_embedding, context_vector)
```

The LSTM now receives both the previous word embedding AND a fresh, dynamically focused summary of the source sentence at every single step.

This entire process — scoring, softmax, weighted sum — repeats freshly at every decoder timestep.

#### Why "Additive"?

Bahdanau attention is called additive because it adds the transformed decoder state and encoder state together (`W₁·s + W₂·hᵢ`) before passing through tanh. This distinguishes it from Luong (multiplicative) attention which uses a dot product instead.

---

## Architecture

### Vanilla Seq2Seq Architecture

```
┌─────────────────────────────────────────────────────┐
│                     ENCODER                         │
│                                                     │
│  "A"  →  Embedding  →  LSTM  →  h1                 │
│  "black" → Embedding → LSTM  →  h2                 │
│  "dog" →  Embedding  →  LSTM  →  h3                │
│  "runs" → Embedding  →  LSTM  →  h4, c4            │
│                                    ↓                │
│                           (h4, c4) only             │
└─────────────────────────────────────────────────────┘
                              ↓
                    Single Context Vector
                              ↓
┌─────────────────────────────────────────────────────┐
│                     DECODER                         │
│                                                     │
│  <sos> → Embedding → LSTM(h4,c4) → Linear → "Ein"  │
│  "Ein" → Embedding → LSTM →  Linear → "schwarzer"  │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

### Attention Seq2Seq Architecture

```
┌─────────────────────────────────────────────────────┐
│                     ENCODER                         │
│                                                     │
│  "A"    → Embedding → LSTM → h1 ──────────────┐    │
│  "black"→ Embedding → LSTM → h2 ──────────────┤    │
│  "dog"  → Embedding → LSTM → h3 ──────────────┤    │
│  "runs" → Embedding → LSTM → h4, c4 ──────────┤    │
│                                                ↓    │
│                              ALL hidden states kept │
└─────────────────────────────────────────────────────┘
                         ↓              ↓
                   (h4,c4) init    [h1,h2,h3,h4]
                         ↓              ↓
┌─────────────────────────────────────────────────────┐
│                ATTENTION DECODER                    │
│                                                     │
│  At each step t:                                    │
│    current decoder state s                          │
│         ↓                                          │
│    score each hᵢ: eᵢ = V·tanh(W₁·s + W₂·hᵢ)      │
│         ↓                                          │
│    softmax → attention weights α                   │
│         ↓                                          │
│    context = Σ αᵢ·hᵢ                              │
│         ↓                                          │
│    concat(embedding, context) → LSTM → predict     │
│         ↓                                          │
│    repeat with updated s                           │
└─────────────────────────────────────────────────────┘
```

---

## Project Pipeline

### 1. Data Loading
Used the `bentrevett/multi30k` dataset from HuggingFace — 29,000 English-German sentence pairs for training, with separate validation and test splits.

### 2. Tokenization
Custom tokenization using spaCy's English (`en_core_web_sm`) and German (`de_core_news_sm`) pipelines rather than relying on pretokenized data.

```python
def tokenize_en(text):
    return [token.text for token in en_nlp.tokenizer(text)]

def tokenize_de(text):
    return [token.text for token in de_nlp.tokenizer(text)]
```

### 3. Vocabulary Construction
Built vocabularies from scratch using `Counter` — words appearing more than once are included, words appearing only once are mapped to `<unk>`. Special tokens reserved at indices 0-3:

```
<unk> = 0
<pad> = 1
<sos> = 2
<eos> = 3
```

Vocab sizes computed as `max(vocab.values()) + 1` to ensure embedding tables are correctly sized.

### 4. Numericalization
Each token mapped to its vocabulary index. Unknown words map to 0.

### 5. DataLoader with Custom Collate
`pad_sequence` used to pad variable-length sentences to the same length within each batch. `<sos>` and `<eos>` tokens injected at the start and end of every sentence. Output shape: `[seq_len, batch_size]` for LSTM compatibility.

### 6. Model Training
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with `ignore_index=1` (ignores padding tokens)
- Gradient clipping: max_norm=1 to prevent exploding gradients
- Teacher forcing ratio: 0.5
- Epochs: 25
- TensorBoard logging for loss tracking

### 7. Evaluation
BLEU score computed on 500 held-out test sentences using `nltk.translate.bleu_score.corpus_bleu`. Greedy decoding used for inference.

---

## Results

| Model | BLEU Score | Architecture |
|-------|-----------|--------------|
| Vanilla Seq2Seq | 18.0 | LSTM Encoder-Decoder, single context vector |
| Bahdanau Attention | 21.0 | LSTM + dynamic attention over all encoder states |

**Sample Translations (Attention Model):**

| English | Model Output | 
|---------|-------------|
| A dog is running in the park | Ein Hund läuft im Sand . |
| A man is playing football | spielt in . |
| A woman is reading a book | Frau hält ein Mikrofon . |

---

## Installation

```bash
git clone https://github.com/GarvGupta25/your-repo-name
cd your-repo-name
pip install torch torchtext datasets spacy nltk
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

---

## Usage

Run the notebook cells in order:

1. Data loading and preprocessing
2. Vocabulary construction
3. Vanilla Seq2Seq training
4. Attention Seq2Seq training
5. BLEU evaluation and translation

To translate a sentence with the trained attention model:

```python
translate("A dog is running in the park", en_vocab, de_vocab, attention_model, device)
```

---

## File Structure

```
├── translator.ipynb       # Main notebook with all code
├── attention_model.pt     # Saved attention model weights
├── vanilla_model.pt       # Saved vanilla model weights
├── en_vocab.pkl           # English vocabulary
├── de_vocab.pkl           # German vocabulary
└── README.md
```

---

## What's Next

- **Beam Search Decoding** — replace greedy decoding with beam search (keep top-k candidates at each step) for higher BLEU scores
- **Transformer Architecture** — replace LSTM with self-attention entirely, enabling full parallelization and better long-range dependency modeling
- **Deployment** — FastAPI REST endpoint hosted on Hugging Face Spaces for live inference
- **Luong Attention** — implement multiplicative attention and compare against Bahdanau

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks.
- Vaswani, A. et al. (2017). Attention Is All You Need.
- Dataset: bentrevett/multi30k via HuggingFace Datasets
