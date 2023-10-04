from typing import List, Literal, TypedDict

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def format_tokens(dialogs: List[Dialog]):
    prompts = []
    for dialog in dialogs:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        content: List[str] = [
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
            for prompt, answer in zip(dialog[::2], dialog[1::2])
        ]
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        content.append(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompts.append(content)
    return prompts


# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model


class LLaMABot:
    def __init__(
        self,
        device,
        model_path: str = None,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=256,  # The maximum numbers of tokens to generate
        min_new_tokens: int = 0,  # The minimum numbers of tokens to generate
        seed: int = None,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
    ):
        if seed is not None:
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

        self.model = load_model(model_path, quantization)
        if peft_model:
            self.model = load_peft_model(self.model, peft_model)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def build_dialogs(self, text):
        text = [text]
        dialogs = format_tokens(text)
        return dialogs

    def answer(self, chats):
        tokens = []
        for chat in chats:
            tokens.append(
                sum(
                    [
                        self.tokenizer.encode(
                            content,
                        )
                        for content in chat
                    ],
                    [],
                )
            )
        tokens = torch.tensor(tokens).long()
        tokens = tokens.to(self.device)
        outputs = self.model.generate(
            tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
            use_cache=self.use_cache,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
        )

        output_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

        return output_text.strip()

    @torch.no_grad()
    def __call__(self, text):
        dialogs = self.build_dialogs(text)

        output = self.answer(dialogs)
        skip_len = sum(len(content) for content in dialogs[0]) + 2

        # response = output
        response: str = output[skip_len:]

        return response.strip()

    def to(self, device):
        pass


def test():
    bot = LLaMABot(
        model_path="model_zoo/llama2/Llama-2-13b-chat-hf",
        device="cuda:0",
        max_new_tokens=512,
        quantization=True,
    )

    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            },
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            },
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]

    for dialog in dialogs:
        result = bot(dialog)
        print(result)
        break


if __name__ == "__main__":
    test()
