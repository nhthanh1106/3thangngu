# DeBERTa-v3-large fact-grounding checkpoint

- `model.safetensors` is the fine-tuned model weight file downloaded from the completed Kaggle training output. File size: 1,740,304,440 bytes.
- `config.json` records the checkpoint architecture and the two class labels observed in that Kaggle artifact.
- `spm.model` and `tokenizer_config.json` come from the public `microsoft/deberta-v3-large` base model repository used for fine-tuning. The runtime can rebuild the fast tokenizer from SentencePiece if `tokenizer.json` is absent.
- `added_tokens.json` and `special_tokens_map.json` preserve the mask and special-token mapping.

Load with Transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint_dir = "fact_grounding_reproduction/checkpoint"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
```

The expected label mapping is `0 = NOT-ANSWERABLE`, `1 = ANSWERABLE`. The classifier expects the same formatted `(sub-question, retrieved passages)` text used by the training notebook.
