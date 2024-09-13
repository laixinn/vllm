**Deployment Guideline**
1. Please down model checkpoint and tokenizer from [google drive](https://drive.google.com/file/d/1eQJ_2uW2Wjfey-0Y6hHASKDyRLPCm5uX/view?usp=sharing), and put them in into `vllm/core/fixation_model/`. The file tree should end up with:
```bash
vllm/core/fixation_model/
|-- bert_tokenizer
|   |-- bert_config.json
|   |-- bert_model.ckpt.data-00000-of-00001
|   |-- bert_model.ckpt.index
|   |-- config.json
|   |-- pytorch_model.bin
|   `-- vocab.txt
|-- eye_attention.py
|-- eye_movement.py
|-- eyettention_model_TECO_n.pth
`-- length_predictor.py
2. To avoid GPU memory overflow, the fixation model runs on 'cuda:1'. Thus deploying this system requires dual gpus on one machine.