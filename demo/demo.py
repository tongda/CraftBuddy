from transformers import BertConfig, BertLMHeadModel, Blip2QFormerConfig, Blip2QFormerModel, InstructBlipQFormerConfig, InstructBlipQFormerModel

import torch
import torch.nn as nn
from craft_buddy.models.qformer import QFormer
from craft_buddy.models.vit import PreTrainViT

import decord
import einops
import numpy as np
import torchvision.transforms.functional as F

import transformers

from craft_buddy.models.vqformer import VideoQFormer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


vit = PreTrainViT.from_blip2_vit_ckpt("ckpt/eva-vit-g/eva_vit_g.pth")
vit = vit.to("cuda:0")
blip2_ckpt = torch.load("ckpt/instruct-blip/instruct_blip_vicuna7b_trimmed.pth", map_location="cpu")['model']
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

frame_qformer = QFormer.from_blip2_ckpt(blip2_ckpt)
with torch.cuda.amp.autocast(dtype=torch.float16):
    img_embs = vit(images).last_hidden_state
    torch.save(images, "img_embs_1.pt")
    print(images[:, :, 100, 100])
    print("img_embs", img_embs)
    instructions = [f"This frame is sampled at {i / vr.get_avg_fps():.1f} seconds." for i in indices]
    frame_embeds = frame_qformer(img_embs, instructions).last_hidden_state
    print(frame_embeds)

    video_qformer = VideoQFormer(max_frame_pos=96, vision_width=frame_qformer.hidden_size)
    print(video_qformer(frame_embeds))

# tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

# blip2_qformer = Blip2QFormerModel(Blip2QFormerConfig()).to("cuda:0")
# query_tokens = nn.Parameter(torch.zeros(1, 32, blip2_qformer.config.hidden_size))
# query_tokens = query_tokens.expand(img_embs.shape[0], -1, -1).to("cuda:0")

# instructblip_qformer = InstructBlipQFormerModel(InstructBlipQFormerConfig()).to("cuda:0")

# input_ids = tokenizer("test, test, test.", return_tensors="pt")
# embs = instructblip_qformer.embeddings(input_ids["input_ids"], query_embeds=query_tokens).to("cuda:0")

# print(instructblip_qformer(input_ids["input_ids"].to("cuda:0"), query_embeds=query_tokens, encoder_hidden_states=img_embs.to("cuda:0")))

# mask = torch.cat(
#     [
#         torch.ones(query_tokens.shape[:-1], dtype=torch.long).to("cuda:0"), 
#         input_ids["attention_mask"].to("cuda:0")
#     ], dim=-1)

# with torch.cuda.amp.autocast():
#     out = blip2_qformer(embs, mask, encoder_hidden_states=img_embs, query_length=query_tokens.shape[1])
# print(out)