# dixi-InferTriggerWords
Try to infer the trigger words with a given fine-tuned diffusion model and a given pre-trained diffusion model
## LoRA
### The code for fine-tuning DreamBooth
```
git clone https://github.com/huggingface/peft
cd peft/examples/lora_dreambooth
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
bash fine-tune.sh
```
You will get the fine-tuned diffusion model.

```train_dreambooth.py``` is the original training code provided by Huggingface. To run this file, I also provide a bash file (```train_dreambooth.sh```) for the command to run the code. To train a stable-diffusion v2.1 file, one need about 16GB, recommended on CUDA.

