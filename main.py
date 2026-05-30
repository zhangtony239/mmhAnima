import torch
import numpy as np
from random import randint

# --- 配置区 ---
ckpt_path = "miaomiaoRealskin_anima10.safetensors"
output_path = "C:\\Users\\Tony\\Downloads\\"
negative_prompt = "worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,"
hotwords = {
    'AIRKI': '1girl,white hair,blue eyes,cat ears',
    }
# --- 配置区 ---

# TODO

pipe = pipe.to("xpu")

# 设置内存格式为 channels_last
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.decoder.to(memory_format=torch.channels_last)

MAX_SEED = np.iinfo(np.int32).max

def draw(prompt,seed):

    print(f"Current seed: {seed}")

    image = pipe(
        # TODO
    ).images[0] # pyright: ignore[reportAttributeAccessIssue]
    
    image.save(f"{output_path}{seed}.png")

if __name__ == "__main__":
    while True:
        prompt = input("Prompt: ").strip()
        if prompt in ['Q','q','exit']:
            break
        if len(prompt.split('seed')) > 1:
            seed = int(prompt.split('seed')[1].split(',')[0][1:])
            prompt = prompt.split('seed')[0] + ','.join(prompt.split('seed')[1].split(',')[1:])
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompt.split(',')]
        processed_tags = [hotwords[t.upper()] if t.upper() in hotwords else t for t in prompt_tags]
        prompt = ",".join(processed_tags)
        draw(prompt, seed)