import torch
import numpy as np
from random import randint

# 猴子补丁：让 diffsynth 支持 Intel XPU 设备类型解析，防止 empty_cache 报错
import diffsynth.core.device.npu_compatible_device as npu_dev
import diffsynth.core as core
import diffsynth.diffusion.base_pipeline as base_pipe

def patched_parse_device_type(device):
    if isinstance(device, str):
        if device.startswith("cuda"):
            return "cuda"
        elif device.startswith("npu"):
            return "npu"
        elif device.startswith("xpu"):
            return "xpu"
        else:
            return "cpu"
    elif isinstance(device, torch.device):
        return device.type

npu_dev.parse_device_type = patched_parse_device_type
core.parse_device_type = patched_parse_device_type
base_pipe.parse_device_type = patched_parse_device_type

from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig # noqa: E402

# --- 配置区 ---
ckpt_path = 'miaomiaoRealskin_anima10.safetensors'
output_path = 'C:\\Users\\Tony\\Downloads\\'
negative_prompt = 'worst quality,bad quality,simple_background,low quality,jpeg artifacts,old,oldest,signature,shiny_skin,bad hands,bad feet,'
hotwords = {
    'AIRKI': '1girl,white hair,blue eyes,cat ears',
    }
vram_config = {
    'offload_dtype': 'disk',
    'offload_device': 'disk',
    'onload_dtype': 'disk',
    'onload_device': 'disk',
    'preparing_dtype': torch.bfloat16,
    'preparing_device': 'xpu',
    'computation_dtype': torch.bfloat16,
    'computation_device': 'xpu',
}
# --- 配置区 ---

pipe = AnimaImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device='xpu',
    model_configs=[
        ModelConfig(model_id='circlestone-labs/Anima', origin_file_pattern='split_files/text_encoders/qwen_3_06b_base.safetensors', **vram_config),
        ModelConfig(model_id='circlestone-labs/Anima', origin_file_pattern='split_files/vae/qwen_image_vae.safetensors', **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id='Qwen/Qwen3-0.6B', origin_file_pattern='./'),
    tokenizer_t5xxl_config=ModelConfig(model_id='stabilityai/stable-diffusion-3.5-large', origin_file_pattern='tokenizer_3/'),
)

# 绕开diffsynth的擅自hash匹配，强制载入第三方DiT
from diffsynth.core.loader import load_model # noqa: E402
from diffsynth.models.anima_dit import AnimaDiT # noqa: E402
from diffsynth.utils.state_dict_converters.anima_dit import AnimaDiTStateDictConverter # noqa: E402
from diffsynth.models.model_loader import ModelPool # noqa: E402

model_pool = ModelPool()
module_map = model_pool.fetch_module_map("diffsynth.models.anima_dit.AnimaDiT", vram_config)

pipe.dit = load_model(
    model_class=AnimaDiT,
    path=ckpt_path,
    config={},
    torch_dtype=vram_config["computation_dtype"],
    device=vram_config["computation_device"],
    state_dict_converter=AnimaDiTStateDictConverter,
    use_disk_map=True,
    vram_config=vram_config,
    module_map=module_map,
)

# 重新检查并更新 VRAM 管理状态
pipe.vram_management_enabled = pipe.check_vram_management_state()

MAX_SEED = np.iinfo(np.int32).max

def draw(prompt,seed):

    print(f'Current seed: {seed}')

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=1536,
        width=1024,
        cfg_scale=4,
        num_inference_steps=30
        )
    
    image.save(f'{output_path}{seed}.png')

if __name__ == '__main__':
    while True:
        prompt = input('Prompt: ').strip()
        if prompt in ['Q','q','exit']:
            break
        if len(prompt.split('seed')) > 1:
            seed = int(prompt.split('seed')[1].split(',')[0][1:])
            prompt = prompt.split('seed')[0] + ','.join(prompt.split('seed')[1].split(',')[1:])
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompt.split(',')]
        processed_tags = [hotwords[t.upper()] if t.upper() in hotwords else t for t in prompt_tags]
        prompt = ','.join(processed_tags)
        draw(prompt, seed)
