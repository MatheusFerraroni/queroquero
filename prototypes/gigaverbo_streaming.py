from transformers import GenerationConfig, TextGenerationPipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset

ds = load_dataset(
    "Polygl0t/gigaverbo-v2",
    "default",
    split="train",
    streaming=True,
)

ds = ds.filter(lambda x: x["edu_int_score"] >= 4)

print(ds)

# # Specify the model and tokenizer
# model_id = "Polygl0t/Tucano2-0.6B-Base"
# revision = "step-160000-end-of-stage-2"  # step-190000

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)

# # Specify the generation parameters as you like
# generation_config = GenerationConfig(
#     **{
#     "do_sample": True,
#     "max_new_tokens": 150,
#     "renormalize_logits": True,
#     "repetition_penalty": 1.2,
#     "temperature": 0.1,
#     "top_k": 50,
#     "top_p": 1.0,
#     "use_cache": True,
#   }
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = TextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, device=device)

# # Generate text
# prompt = "# A floresta da Amazônia: um lugar de Magia\n\n"
# completion = generator(prompt, generation_config=generation_config)
# print(completion[0]['generated_text'])
