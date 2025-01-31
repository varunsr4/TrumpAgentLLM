import autogen
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel, PeftConfig

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and apply LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "trump_lora_model/final",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Merge LoRA weights with base model for better inference performance
model = model.merge_and_unload()

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    device_map="auto"
)

# Trump's system prompt
TRUMP_PROMPT = """You are Donald Trump, the 45th President of the United States. 
Respond to messages in your characteristic style, using your typical mannerisms and speech patterns."""

# Configure the local LLM endpoint
config_list = [{
    "model": "local",
    "base_url": None,  # Local model
    "api_key": "NOT_NEEDED",  # Not needed for local model
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "max_tokens": 256,
    "model": model_name,
    "functions": [{
        "name": "generate",
        "description": "Generate a response using local DeepSeek model",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to respond to"
                }
            },
            "required": ["message"]
        }
    }]
}

def generate_response(message):
    full_prompt = f"{TRUMP_PROMPT}\n\nInterviewer: {message}\nTrump:"
    response = pipe(full_prompt)[0]['generated_text']
    # Extract only the response part (after "Trump:")
    response = response.split("Trump:")[-1].strip()
    return response

# Create the Trump agent
trump_agent = autogen.ConversableAgent(
    name="Trump",
    system_message=TRUMP_PROMPT,
    llm_config=llm_config,
    human_input_mode="NEVER",
    function_map={"generate": generate_response}
)

# Create the user proxy agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=0
)

def main():
    print("🎤 Trump Chatbot initialized with LoRA fine-tuning! Type 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    
    while True:
        # Get user input
        user_message = input("\nYou: ").strip()
        
        # Check for exit command
        if user_message.lower() in ['quit', 'exit']:
            print("\nTrump: Bye bye folks! It was tremendous, really tremendous!")
            break
            
        # Generate response using our custom pipeline
        response = generate_response(user_message)
        print(f"\nTrump: {response}")

if __name__ == "__main__":
    main() 