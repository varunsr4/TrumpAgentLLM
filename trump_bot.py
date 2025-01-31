import autogen
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # You'll need to adjust this based on your Colab setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

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

# Configure the Trump agent
config_list = [{"model": "llama2"}]  # Placeholder config, will use our local pipeline

def generate_response(message):
    full_prompt = f"{TRUMP_PROMPT}\n\nUser: {message}\nTrump:"
    response = pipe(full_prompt)[0]['generated_text']
    # Extract only the response part (after "Trump:")
    response = response.split("Trump:")[-1].strip()
    return response

# Create the Trump agent
trump_agent = autogen.ConversableAgent(
    name="Trump",
    system_message=TRUMP_PROMPT,
    llm_config={"config_list": config_list},
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
    print("🎤 Trump Chatbot initialized! Type 'quit' or 'exit' to end the conversation.")
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