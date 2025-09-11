import numpy as np
import sentencepiece
import openpi.shared.download as download

path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with path.open("rb") as f:
    _tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

tokenized_prompt = [[2, 1065, 48068, 1280, 3741, 108, 0, 0, 0]]
print(_tokenizer.decode(tokenized_prompt))