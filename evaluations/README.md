## Zero-shot classification with OpenCLIP models


```shell
python3 zeroshot_cls.py --dataset "evaluation dataset, such as CIFAR10" \
                        --data_path "PATH/TO/THE/DATASET"               \
                        --model "which model to evaluate, see OpenCLIP" \
                        --pretrained "specific weight, see OpenCLIP"    \
                        --threat_model "linf_non_targeted"              \
                        --attacker_name "which attacker to use" 
```

## Image-text retrieval with OpenCLIP models


```shell
python3 eval_retrieval.py --model "which model to evaluate, see OpenCLIP" \ 
                          --pretrained "specific weight, see OpenCLIP"    \
                          --threat_model "linf_non_targeted"              \
                          --attacker_name "which attacker to use" 
```

## Zero-shot classification with ALIGN
```shell
python3 zeroshot_cls.py --dataset "evaluation dataset, such as CIFAR10" \
                        --data_path "PATH/TO/THE/DATASET"               \
                        --threat_model "linf_non_targeted"              \
                        --attacker_name "which attacker to use" 
```

## Evaluate with VLMs

- Ensure that all dependencies are satisfied. The default is `requirements.txt`, but BLIP-2, LLaVA, and MiniGPT-4 may require different versions.
- Review the script-specific arguments for evaluation:
    - `eval_blip2.py`
	- `eval_flamingo.py`
	- `eval_llava.py`
	- `eval_minigpt4.py`
	- `eval_florence2.py`
- VLM evaluation supports:
    - Image Captioning: COCO, Flickr30k
	- Visual Question Answering (VQA): OK-VQA, VizWiz