from transformers import AutoTokenizer, TFAutoModelForCausalLM, pipeline
import gradio as gr

model_name = "microsoft/DialoGPT-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = TFAutoModelForCausalLM.from_pretrained(model_name)

def chat_with_bot(message):
    prompt = (
        "User:   Does money buy happiness?"
        "Bot :   Depends how much money you spend on it ."
        "User:   What is the best way to buy happiness ?"
        "Bot :   You just have to be a millionaire by your early 20s, then you can be happy ."
        "User:   " + message +
        "Bot : "
    )

    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="tf")

    outputs = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.5, top_k=10, top_p=0.95)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    bot_response = generated_text.split("Bot :")[-1].strip()
    
    return bot_response


demo = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="Mesajınızı yazın"),
    outputs=gr.Textbox(label="Bot Yanıtı"),
    title="DialoGPT Chatbot",
    description="Microsoft DialoGPT ile sohbet edin!"
)

if __name__ == "__main__":
    demo.launch()





