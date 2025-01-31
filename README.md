# TrumpAgentLLM

A simple Donald Trump chatbot powered by Llama 3 and Microsoft's AutoGen framework. The chatbot engages in conversation with users, mimicking Donald Trump's communication style.

## Overview
This project implements a basic conversational agent that:
- Uses AutoGen for the agent framework
- Leverages a small Llama 3 model as the core LLM
- Provides a simple console interface for user interaction

## Running on Google Colab

1. Create a new Google Colab notebook
2. Clone this repository:
   ```bash
   !git clone https://github.com/[your-username]/TrumpAgentLLM.git
   %cd TrumpAgentLLM
   ```

3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```

4. Run the chatbot:
   ```bash
   !python trump_bot.py
   ```

## Requirements
- Python 3.8+
- PyTorch
- AutoGen
- Transformers
- Google Colab with A100 GPU

## Usage
After running the script, you can interact with the Trump chatbot through the console:
1. Type your message and press Enter
2. The chatbot will respond in Trump's style
3. Type 'quit' or 'exit' to end the conversation

## Note
This is a basic implementation without fine-tuning. The Trump-like responses are generated using a simple system prompt.