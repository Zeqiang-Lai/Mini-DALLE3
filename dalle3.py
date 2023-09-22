import gradio as gr
import openai
from diffusers import DiffusionPipeline
import torch
import re


pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# variation =

system_message = {
    'role': 'system',
    'content': open('prompt.txt', 'r').read().strip()
}

pattern = r'<image>(.*?)<\/image>'


def user(content):
    return {'role': 'user', 'content': content}


def ai(content):
    return {'role': 'assistant', 'content': content}


def run(messages):
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    response = result['choices'][0]['message']['content']
    return response


def extract_image(message):
    matches = re.findall(pattern, message)
    for match in matches:
        return match.strip()
    return None


def remove_image(message):
    return re.sub(pattern, '', message)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def bot(history, state):
    state['messages'].append(user(history[-1][0]))
    output = run(state['messages'])
    state['messages'].append(ai(output))

    response = output

    image_prompt = extract_image(output)
    if image_prompt is not None:
        print(image_prompt)
        image = pipeline(image_prompt).images[0]
        image_filename = 'image.png'
        image.save(image_filename)
        history[-1][1] = remove_image(response)
        history.append((None, (image_filename,)))
    else:
        history[-1][1] = response

    return history, state


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div align='center'> <h1>Mini-DALLE3 </h1> </div>
        <p align="center"> Replication of Next Generation Text to Image Model. </p>
        """,
    )
    
    state = gr.State({'messages': [system_message], 'images': []})
    chatbot = gr.Chatbot(
        [],
        bubble_full_width=False,
        height=600,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    gr.Examples(
        ['I have read a story where it talks about an "astronaut riding a horse" -- What does it look like ?',
         'Can I see more like this ?',
         'Can you make the horse run on the grassland ?',
         'Looks great ! Could you tell me why this image is strange ?',
         'Cool ! Could you make some sticker ?'],
        inputs=[txt],
        label='Examples',
    )

demo.queue()
if __name__ == "__main__":
    demo.launch()
