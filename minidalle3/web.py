import gradio as gr
import logging
import fire
import uuid
import os
import datetime
import json

from .model import MiniDALLE3


logging.basicConfig(
    filename="minidalle3.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)

logger = logging.getLogger(__file__)
SAVE_DIR = "saved"

user_quota = {}
if os.path.exists("user_quota.json"):
    user_quota = json.load(open("user_quota.json", "r"))
MAX_QUOTA_PER_USER = 30


def user(content):
    return {"role": "user", "content": content}


def ai(content):
    return {"role": "assistant", "content": content}


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def bot(history, state, allowed, request: gr.Request):
    current_datetime = datetime.datetime.now()
    current_date_as_string = current_datetime.strftime("%Y-%m-%d")
    save_dir = os.path.join(SAVE_DIR, request.client.host, current_date_as_string, state["id"])
    os.makedirs(save_dir, exist_ok=True)

    # track quota
    if current_date_as_string not in user_quota:
        user_quota[current_date_as_string] = {}
    if request.client.host not in user_quota[current_date_as_string]:
        user_quota[current_date_as_string][request.client.host] = MAX_QUOTA_PER_USER
    quota = user_quota[current_date_as_string][request.client.host]
    if quota <= 0:
        history[-1][1] = f"Sorry, you have reach the limit of request per day ({MAX_QUOTA_PER_USER})."
        return history, state

    state["messages"].append(user(history[-1][0]))
    raw_output, response = state["model"].ask(state["messages"], state["images"])
    state["messages"].append(ai(raw_output))

    if response is not None:
        image_filename = f"{uuid.uuid4()}.png"
        image_filename = os.path.join(save_dir, "img", image_filename)
        image = response.image
        os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
        image.save(image_filename)
        state["images"].append(image)
        state["image_prompts"].append(response.image_prompt)
        history[-1][1] = response.response
        history.append((None, (image_filename,)))
    else:
        history[-1][1] = raw_output

    if allowed:
        json.dump(history, open(os.path.join(save_dir, "chat.json"), "w"), indent=4)

    user_quota[current_date_as_string][request.client.host] -= 1
    json.dump(user_quota, open("user_quota.json", "w"), indent=4)
    return history, state


def on_select_image(evt: gr.SelectData, state):
    return state["image_prompts"][evt.index]


def main(llm="gpt3.5", port=10049, prompt_path=None, max_users=1, share=False):
    model = MiniDALLE3(llm, prompt_path=prompt_path)

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <div align='center'> <h1> Mini DALL•E 3 </h1> </div>
            <p align="center"> Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models. </p>
            <p align="center"><a href="https://github.com/Zeqiang-Lai/MiniDALLE-3">Github</a> | <a href="minidalle3.github.io/static/minidalle3.pdf">Paper</a> | <a href="minidalle3.github.io">Project Page</a></p>
            """,
        )

        state = gr.State({"messages": [model.system_message], "images": [], "image_prompts": [], "model": model, "id": str(uuid.uuid4())})

        with gr.Tab("Mini DALLE3"):
            chatbot = gr.Chatbot([], bubble_full_width=False, height=600, avatar_images=["assets/man.png", "assets/bot.png"])
        with gr.Tab("Prompt History"):
            prompt = gr.TextArea(label="Prompt", placeholder="Click the image below to see the prompt.")
            gallery = gr.Gallery(label="Generated Image")

        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
                container=False,
                scale=4,
            )
            btn = gr.Button("Submit", variant="primary", min_width=50, scale=1)

        gr.Examples(
            [
                'I have read a story where it talks about an "astronaut riding a horse" -- What does it look like ?',
                "Can I see more like this ?",
                "Can you make the horse run on the grassland ?",
                "Looks great ! Could you tell me why this image is strange ?",
            ],
            inputs=[txt],
            label="Examples",
        )
        allowed = gr.Checkbox(value=True, label="Will you allow us to save the chat history for future development?")

        txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(bot, [chatbot, state, allowed], [chatbot, state])
        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

        btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(bot, [chatbot, state, allowed], [chatbot, state]).then(
            lambda: gr.update(interactive=True), None, [txt], queue=False
        ).then(lambda state: state["images"], [state], [gallery])
        gallery.select(on_select_image, [state], [prompt])

    demo.queue(concurrency_count=max_users).launch(server_port=port, server_name="0.0.0.0", share=share)


if __name__ == "__main__":
    fire.Fire(main)
