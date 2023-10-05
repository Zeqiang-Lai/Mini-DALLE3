<p align="center">
<a href="#">
    <img src="https://github.com/Zeqiang-Lai/Mini-DALLE3/assets/26198430/11856c34-b5ef-4665-8cb9-8a6e366ae238" alt="minidalle3" width="19%">
    </a> &ensp; 
</p>

<p align="center">
    <b>Interactive Text to Image Generation </b>
</br>
<a href="https://light.princeton.edu/publication/delta_prox/">Paper</a> •
<a href="https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/notebooks">Demo</a> •
<a href="https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/examples">Project page</a> 
</p>


https://github.com/Zeqiang-Lai/Mini-DALLE3/assets/26198430/78250401-de79-4878-97a7-201a0a2ab687

> An experimental attempt to obtain the interactive and interleave text to image and text to text experience of [DALL•E 3](https://openai.com/dall-e-3) and [ChatGPT](https://openai.com/chatgpt).

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
python -m minidalle3.serves
```

## TODO

- [x] Support generating image interleaved in the conversations.
- [ ] Support generating multiple images at once.
- [ ] Support selecting image.


## Citation

If you find this repo helpful, please consider citing us.

```bibtex
@misc{minidalle3,
    author={Zeqiang Lai, Wenhai Wang},
    title={Mini-DALLE3: Interactive Text to Image Generation by Prompting Large Language Models},
    year={2023},
    url={https://github.com/Zeqiang-Lai/Mini-DALLE3},
}
```

## Acknowledgement

[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) • [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FZeqiang-Lai%2FMini-DALLE3&countColor=%23263759&style=flat)
