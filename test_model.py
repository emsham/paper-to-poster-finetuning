                                              
from transformers import AutoModelForCausalLM, AutoTokenizer                      
from peft import PeftModel                                                        
import torch                                                                      
                                                                                
# Load model                                                                      
print("Loading model...")                                                         
base_model = AutoModelForCausalLM.from_pretrained(                                
  "mistralai/Mistral-7B-Instruct-v0.2",                                         
  torch_dtype=torch.bfloat16,                                                   
  device_map="auto"                                                             
)                                                                                 
model = PeftModel.from_pretrained(base_model, "./poster-mistral-lora")            
tokenizer = AutoTokenizer.from_pretrained("./poster-mistral-lora")                
                                                                                
# Sample prompt                                                                   
instruction = "Generate an academic poster based on the following paper content." 
paper_text = """Title: Deep Learning for Image Classification                     
Abstract: We present a novel CNN architecture that achieves state-of-the-art      
results on ImageNet. Our method uses residual connections and attention mechanisms
to improve accuracy by 5% over previous methods."""                              
                                                                                
prompt = f"<s>[INST] {instruction}\n\n{paper_text} [/INST]"                       
                                                                                
# Generate                                                                        
print("Generating...")                                                            
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)                  
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7,          
do_sample=True)                                                                   
result = tokenizer.decode(outputs[0], skip_special_tokens=True)                   
                                                                                
print("\n" + "="*60)                                                              
print("OUTPUT:")                                                                  
print("="*60)                                                                     
print(result.split("[/INST]")[-1].strip())                                        