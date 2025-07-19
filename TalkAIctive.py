"""
Name: TalkAIctive
Description: Interactive OpenAI and Gradio App. Context-Aware and Streaming.

Instruction:
        1. Download TalkAIctive: https://github.com/VelansRex/TalkAIctive/releases/download/Use/TalkAIctive.exe
        2. Run File: TalkAIctive.exe
        3. Repo: https://github.com/VelansRex/TalkAIctive.git

About App:
App uses OpenAI GPT-4o-mini model and Gradio UI.
"""

import openai
from dotenv import load_dotenv
import gradio as gr
import os
import sys
import threading
import time
import signal

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Context-Aware
chat_history = []

# Set up history/context
MAX_HISTORY = 2

def query(question):
	global chat_history
	chat_history.append({"role": "user", "content": question})

	# Limiting context to 4 messages
	if len(chat_history) > MAX_HISTORY * 2:
		chat_history = chat_history[-MAX_HISTORY * 2:]

	response = openai.ChatCompletion.create(
		model="gpt-4o-mini",
		messages=chat_history,
		temperature=0.2,
		max_tokens=1000,
		stream=True
	)

	answer_chunks = []
	streamed_text = ""

	# Generator streaming chunks - Stream + Collect + Yield = Live typing effect
	for chunk in response:
		content = chunk["choices"][0]["delta"].get("content")
		finish_reason = chunk["choices"][0].get("finish_reason")
		if content:
			answer_chunks.append(content)
			streamed_text += content
			yield streamed_text
		if finish_reason is not None:
			full_answer = "".join(answer_chunks)
			chat_history.append({
				"role": "assistant",
				"content": full_answer,
			})
			yield full_answer
			# The generator ends here

# Gradio Interface
theme = gr.themes.Base(
).set(
    body_background_fill="white",
    body_text_color="black",
    block_background_fill="white",
    button_primary_background_fill="#3cb371",  # mediumseagreen
    button_primary_background_fill_hover="#2e8b57",  # seagreen
    button_primary_text_color="white",
    input_background_fill="white",
    input_border_color="#3cb371",
)

# Patch sys.stdout to avoid NoneType error
if sys.stdout is None:
	sys.stdout = open(os.devnull, 'w')

def shutdown():
	os.kill(os.getpid(), signal.SIGTERM)

with gr.Blocks(theme=theme, title="TalkAIctive", css="body { font-family: sans-serif; }") as interface:
	gr.Image(value=os.path.join(getattr(sys, '_MEIPASS', os.path.abspath('.')), "TalkAIctive.png"), show_label=False)

	with gr.Row():
		with gr.Column(scale=1):
			gr.Markdown("### Interactive OpenAI and Gradio App. Context-Aware and Streaming.")

	with gr.Row():
		with gr.Column(scale=1):
			inp = gr.Textbox(
				label="Talk Active:",
				lines=5,
				placeholder="Your prompt",
				info="AI will remember up to 4 messages."
			)
			send = gr.Button("Send", variant="primary")
			shutdown_btn = gr.Button("Shutdown Server", variant="stop")
			shutdown_btn.click(fn=shutdown, inputs=[], outputs=[])

		with gr.Column(scale=1):
			out = gr.Textbox(
				label="Artificial Intelligence:",
				show_copy_button=True
			)

		send.click(fn=query, inputs=inp, outputs=out)

def start_gradio():
	interface.launch(server_name="127.0.0.1", inbrowser=True)

gradio_thread = threading.Thread(target=start_gradio)
gradio_thread.start()