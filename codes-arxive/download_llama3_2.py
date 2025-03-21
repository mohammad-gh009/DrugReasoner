
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="/home/u111169/wrkdir/mgh-project/models")