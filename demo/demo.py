import torch
from craft_buddy.models.qformer import QFormer
from craft_buddy.models.vit import PreTrainViT

import decord
import einops
import numpy as np
import torchvision.transforms.functional as F

from craft_buddy.models.vqformer import VideoQFormer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


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

# frame_qformer.query_tokens.data.copy_(timechat_ckpt["query_tokens"])
with torch.cuda.amp.autocast(dtype=torch.float16):
    img_embs = vit(images).last_hidden_state
    instructions = [f"This frame is sampled at {i / vr.get_avg_fps():.1f} second." for i in indices]
    frame_embeds = frame_qformer(img_embs, instructions).last_hidden_state
    print(frame_embeds)

    video_qformer = VideoQFormer.from_timechat(timechat_ckpt)
    video_token = video_qformer(frame_embeds)
    
    print(video_token)