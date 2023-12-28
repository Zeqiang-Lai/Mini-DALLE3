<p align="center">
<a href="https://minidalle3.github.io/">
    <img src="https://github.com/Zeqiang-Lai/Mini-DALLE3/assets/26198430/9594f306-cc1a-4a92-bca2-0c64e8daf9c9" alt="minidalle3" width="19%">
    </a> &ensp; 
</p>

<p align="center">
<a href="http://arxiv.org/abs/2310.07653">Technical Report</a> •
<a href="https://minidalle3.github.io/">Project page</a> •
<a href="#">Demo (Temporarily Unavailable)</a>
</p>


https://github.com/Zeqiang-Lai/Mini-DALLE3/assets/26198430/5b6c0a0c-ebbf-48db-981e-f97d542a38b4



![teaser4](https://github.com/Zeqiang-Lai/Mini-DALLE3/assets/26198430/1f17e3c3-6804-4c4e-9266-e902ecedeae8)


> An experimental attempt to obtain the interactive and interleave text-to-image and text-to-text experience of [DALL•E 3](https://openai.com/dall-e-3) and [ChatGPT](https://openai.com/chatgpt).

## Try Yourself 🤗 

- Download the [checkpoint](https://huggingface.co/h94/IP-Adapter) and save it as following 
```bash
checkpoints
   - models
   - sdxl_models
```

- run the following commands, and you will get a gradio-based web demo.

```bash
export OPENAI_API_KEY="your key"
python -m minidalle3.web 
```

- To use other LLM rather than ChatGPT, such as `baichuan`.

```bash
python -m minidalle3.llm.baichuan
export OPENAI_API_BASE="http://0.0.0.0:10039/v1"
python -m minidalle3.web
```

>  `chatglm`, `baichuan`, `internlm` are tested.
> llama have not supported yet. qwen is not tested.

## TODO

- [x] Support generating image interleaved in the conversations.
- [ ] Support generating multiple images at once.
- [ ] Support selecting image.
- [ ] Support refinement.
- [ ] Support prompt refinement/variation.
- [ ] Instruct tuned LLM/SD.


## Citation

If you find this repo helpful, please consider citing us.

```bibtex
@misc{minidalle3,
    author={Lai, Zeqiang and Zhu, Xizhou and Dai, Jifeng and Qiao, Yu and Wang, Wenhai},
    title={Mini-DALLE3: Interactive Text to Image by Prompting Large Language Models},
    year={2023},
    url={https://github.com/Zeqiang-Lai/Mini-DALLE3},
}
```

## Acknowledgement

[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) • [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FZeqiang-Lai%2FMini-DALLE3&countColor=%23263759&style=flat)
