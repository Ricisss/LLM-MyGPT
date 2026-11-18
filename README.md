# LLM MyGPT

A small GPT-style language model project trained on Czech news text.

## Scripts

| Script | What it does |
| --- | --- |
| `get_dataset.py` | Downloads the Czech news dataset and combines it into `dataset.txt`. |
| `tokenizer.py` | Trains a 16,000-token BPE tokenizer and saves it as `tokenizer.json`. |
| `train.py` | Trains the GPT model locally with a CUDA GPU and saves checkpoints. |
| `modal_train.py` | Runs the same training job in a cloud GPU container using Modal. |
| `infer.py` | Loads `my_custom_gpt.pt` and generates text from a sample prompt. |

## Typical workflow

1. Run `get_dataset.py` to create `dataset.txt`.
2. Run `tokenizer.py` to create `tokenizer.json`.
3. Run `train.py` locally, or `modal_train.py` to train on Modal.
4. Run `infer.py` to test the completed model.
