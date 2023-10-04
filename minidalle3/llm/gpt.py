import openai


def chat(messages, model="gpt-3.5-turbo"):
    result = openai.ChatCompletion.create(
        model=model,
        # model="gpt-4",
        messages=messages,
        temperature=0,
    )
    response = result['choices'][0]['message']['content']
    return response
