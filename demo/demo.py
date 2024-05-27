import torch
from craft_buddy.models.llm import LLM
from craft_buddy.models.qformer import QFormer
from craft_buddy.models.vit import PreTrainViT

import decord
import einops
import numpy as np
import torchvision.transforms.functional as F
import torch.nn as nn
from transformers import StoppingCriteria, StoppingCriteriaList

from craft_buddy.models.vqformer import VideoQFormer


vit = PreTrainViT.from_blip2_vit_ckpt("ckpt/eva-vit-g/eva_vit_g.pth")
vit = vit.to("cuda:0")
blip2_ckpt = torch.load("ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth", map_location="cpu")['model']
timechat_ckpt = torch.load("ckpt/timechat/timechat_7b.pth", map_location="cpu")['model']
vit.update_output_layernorm(blip2_ckpt["ln_vision.weight"], blip2_ckpt["ln_vision.bias"])
vit.to_precision(torch.float16)
for name, param in vit.named_parameters():
    param.requires_grad = False
vit = vit.eval()
vit.train = False

decord.bridge.set_bridge('torch')

vr = decord.VideoReader(
    "data/demo.mp4",
    height=224,
    width=224,
)

indices = np.arange(0, len(vr), len(vr) / 96).astype(int).tolist()
print(indices)
images = vr.get_batch([indices])
# images = torch.unsqueeze(images, 0).float()
images = images.float() # 不需要unsqueeze，因为在timechat的原始实现里，并不是batch处理，而是对于每一段视频，单独处理，所以已经没有batch维度，只有time维度
images = einops.rearrange(images, "t h w c ->  t c h w")
images = images / 255.0
mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
images = F.normalize(images, mean, std).cuda(0)

frame_qformer = QFormer.from_blip2_ckpt(timechat_ckpt)
video_qformer = VideoQFormer.from_timechat(timechat_ckpt)

llm = LLM.from_timechat("ckpt/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf", timechat_ckpt, device="cuda:0")

llm_proj = nn.Linear(
    video_qformer.qformer.hidden_size, llm.llm.config.hidden_size, device="cuda:0"
)
llm_proj.weight.data.copy_(timechat_ckpt["llama_proj.weight"])
llm_proj.bias.data.copy_(timechat_ckpt["llama_proj.bias"])

prompt = """
"[INST] <<SYS>>\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n<Video><ImageHere></Video> The video contains 96 frames sampled at 0.0, 0.4, 0.9, 1.4, 1.9, 2.4, 2.8, 3.3, 3.8, 4.2, 4.7, 5.2, 5.7, 6.2, 6.6, 7.1, 7.6, 8.0, 8.5, 9.0, 9.5, 9.9, 10.4, 10.9, 11.4, 11.8, 12.3, 12.8, 13.2, 13.7, 14.2, 14.7, 15.2, 15.6, 16.1, 16.6, 17.0, 17.5, 18.0, 18.5, 19.0, 19.4, 19.9, 20.4, 20.8, 21.3, 21.8, 22.3, 22.8, 23.2, 23.7, 24.2, 24.6, 25.1, 25.6, 26.0, 26.5, 27.0, 27.5, 28.0, 28.4, 28.9, 29.4, 29.8, 30.3, 30.8, 31.3, 31.8, 32.2, 32.7, 33.2, 33.6, 34.1, 34.6, 35.1, 35.6, 36.0, 36.5, 37.0, 37.4, 37.9, 38.4, 38.9, 39.3, 39.8, 40.3, 40.8, 41.2, 41.7, 42.2, 42.6, 43.1, 43.6, 44.1, 44.6, 45.0 seconds.  Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. The output format of each predicted event should be like: 'start - end seconds, event description'. A specific example is : ' 90 - 102 seconds, spread margarine on two slices of white bread in the video' . [/INST]"
"""

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False



with torch.cuda.amp.autocast(dtype=torch.float16):
    img_embs = vit(images).last_hidden_state
    instructions = [f"This frame is sampled at {i / vr.get_avg_fps():.1f} second." for i in indices]
    frame_embeds = frame_qformer(img_embs, instructions).last_hidden_state
    print(frame_embeds)

    video_token = video_qformer(frame_embeds)
    print(video_token)
    
    proj_token = llm_proj(video_token)
    stop_words_ids = [torch.tensor([2]).to("cuda:0")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    output = llm(
        proj_token,
        prompt,
        max_new_tokens=1000,
        stopping_criteria=stopping_criteria,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=0.9,
    )
    
    print(output)