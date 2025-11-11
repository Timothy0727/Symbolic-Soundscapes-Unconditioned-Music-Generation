# AutoComposer — Symbolic, Unconditioned Music Generation

## Notebook Preview
A static HTML version of the notebook is included for easy viewing:
[View workbook.html](https://timothy0727.github.io/Symbolic-Soundscapes-Unconditioned-Music-Generation/)

## Goals
- Learn a distribution over musical events and sample new sequences.
- Compare interpretable baselines on pitch–velocity–duration tokens.
- Output MIDI for listening and simple quantitative evaluation.

## Models
- **Second-Order Markov Chain**: `P(x_t | x_{t-1}, x_{t-2})`
- **Hidden Markov Model (bigram state)**: hidden state `(x_{t-1}, x_t)` with deterministic emission of `x_t`
- **N-gram (n=3)**: `P(x_t | x_{t-2}, x_{t-1})`
- **Polyphonic HMM**: bigram transitions over chord tokens `{pitches}, velocity, duration`

## Data and I/O
- **Input**: `adl-piano-midi/` (dataset of `.mid` files)
- **Outputs**: `output/`
  - `markov_output.mid`
  - `hmm_output.mid`
  - `ngram_output.mid`
  - `hmm_poly_output.mid`


## Repo Layout
```text
.
├── notebooks/
│   └── your_notebook.ipynb
├── workbook.html
├── adl-piano-midi/          # input data (not tracked)
├── output/                  # generated MIDI + plots
├── requirements.txt
├── README.md
└── .gitignore
```

## Environment
Python 3.10+

```bash
pip install -r requirements.txt
```

## Method
### Tokenization
- Monophonic tokens: (pitch, velocity, duration)
- Polyphonic tokens: (frozenset(pitches), velocity, duration)
- Map events to integer IDs and back.

### Training
- Markov: count transitions (x_{t-2}, x_{t-1}) -> x_t and normalize.
- HMM: transitions on hidden bigrams (x_{t-1}, x_t) -> (x_t, x_{t+1}); emit the second element.
- N-gram: count n-gram frequencies and estimate P(next | context).

### Generation
Seed with two tokens, sample stepwise from learned probabilities, detokenize, and write single-track MIDI at 120 BPM, 4/4.

### Evaluation
- Next-token accuracy: argmax prediction vs. ground truth on held-out steps.
- Average log-likelihood per transition: higher (less negative) indicates better fit to training distribution.

## Usage
1. Place dataset:
```text

adl-piano-midi/
  └── Classical/**/*.mid

 ```

 2. Run the notebook to generate files under `output/`.

## Notes
- MIDI parsing uses mido; unreadable files are skipped.
- Writer uses ticks_per_beat=96. The polyphonic writer trims very long sustains.
- For large audio renders, use Git LFS or provide download links.