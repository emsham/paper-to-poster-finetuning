                                                            
from peft import PeftModel                                                        
from transformers import AutoModelForCausalLM, AutoTokenizer                      
import torch                                                                      
                                                                                
# Load and push                                                                   
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2',
torch_dtype=torch.bfloat16)                                                      
model = PeftModel.from_pretrained(model, './poster-mistral-lora')                 
tokenizer = AutoTokenizer.from_pretrained('./poster-mistral-lora')                
                                                                                
# Push to hub (replace with your username/repo name)                              
model.push_to_hub('emsham/poster-mistral-lora')                            
tokenizer.push_to_hub('emsham/poster-mistral-lora')                        
print('Done!')                                                     