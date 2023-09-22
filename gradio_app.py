import gradio as gr
import logging

from minidalle3 import MiniDALLE3


logging.basicConfig(
    filename='minidalle3.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
)

logger = logging.getLogger(__file__)

model = MiniDALLE3()


def user(content):
    return {'role': 'user', 'content': content}


def ai(content):
    return {'role': 'assistant', 'content': content}


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def bot(history, state):
    state['messages'].append(user(history[-1][0]))
    raw_output, response = model.ask(state['messages'], state['images'])
    state['messages'].append(ai(raw_output))

    if response is not None:
        image_filename = 'image.png'
        image = response.image
        image.save(image_filename)
        state['images'].append(image)
        history[-1][1] = response.response
        history.append((None, (image_filename,)))
    else:
        history[-1][1] = raw_output

    return history, state


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div align='center'> <h1> Mini DALL•E 3 </h1> </div>
        <p align="center"> Mini-DALLE3: Interactive Text to Image Generation by Prompting Large Language Models. </p>
        """,
    )

    state = gr.State({'messages': [model.system_message], 'images': []})
    chatbot = gr.Chatbot(
        [],
        bubble_full_width=False,
        height=600,
        avatar_images=['assets/man.png', 'assets/bot.png']
    )

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
            scale=4,
        )
        btn = gr.Button('Submit', variant='primary', min_width=50, scale=1)

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    ).then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    gr.Examples(
        ['I have read a story where it talks about an "astronaut riding a horse" -- What does it look like ?',
         'Can I see more like this ?',
         'Can you make the horse run on the grassland ?',
         'Looks great ! Could you tell me why this image is strange ?',
         'Cool ! Could you make some sticker ?'],
        inputs=[txt],
        label='Examples',
    )

if __name__ == "__main__":
    demo.queue().launch(width='60%')
