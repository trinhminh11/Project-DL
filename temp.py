from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, LlamaForCausalLM, pipeline
import torch

torch.set_grad_enabled(False)

model_path = "meta-llama/Llama-3.2-3B-Instruct"
token = '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'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True, token = token, add_eos_token=True)
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True, token = token)



a = ()