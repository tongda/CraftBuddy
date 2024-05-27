import einops
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from craft_buddy.models.llm import LLM
from craft_buddy.models.qformer import QFormer
from craft_buddy.models.vit import PreTrainViT
from craft_buddy.models.vqformer import VideoQFormer


class DisVLM():
    def __init__(self, vit_ckpt_path: str, blip2_ckpt_path: str, timechat_ckpt_path: str, llm_base_model_path: str, encoder_device: str, llm_device: str):
        blip2_ckpt = torch.load(blip2_ckpt_path, map_location="cpu")['model']
        timechat_ckpt = torch.load(timechat_ckpt_path, map_location="cpu")['model']
        
        self.vit = self._init_vit(vit_ckpt_path, encoder_device, blip2_ckpt)
        self.frame_qformer = QFormer.from_blip2_ckpt(timechat_ckpt, device=encoder_device)
        self.video_qformer = VideoQFormer.from_timechat(timechat_ckpt, device=encoder_device)

        self.llm = LLM.from_timechat(llm_base_model_path, timechat_ckpt, device=llm_device)
        self.llm_proj = self._init_llm_proj(timechat_ckpt, device=llm_device)
        
    def _init_vit(self, vit_ckpt_path: str, vit_device: str, blip2_ckpt: dict):
        vit = PreTrainViT.from_blip2_vit_ckpt(vit_ckpt_path)
        vit = vit.to(vit_device)
        vit.update_output_layernorm(blip2_ckpt["ln_vision.weight"], blip2_ckpt["ln_vision.bias"])
        vit.to_precision(torch.float16)
        for name, param in vit.named_parameters():
            param.requires_grad = False
        vit = vit.eval()
        vit.train = False
        return vit
    
    def _init_llm_proj(self, timechat_ckpt: dict, device: str = "cuda:0"):
        llm_proj = nn.Linear(
            self.video_qformer.qformer.hidden_size, self.llm.llm.config.hidden_size, device=device
        )
        llm_proj.weight.data.copy_(timechat_ckpt["llama_proj.weight"])
        llm_proj.bias.data.copy_(timechat_ckpt["llama_proj.bias"])
        return llm_proj
    
    def encode_video(self, video: torch.Tensor, indices: list[int], fps: float):
        images = images.float() # 不需要unsqueeze，因为在timechat的原始实现里，并不是batch处理，而是对于每一段视频，单独处理，所以已经没有batch维度，只有time维度
        images = einops.rearrange(images, "t h w c ->  t c h w")
        images = images / 255.0
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        images = F.normalize(images, mean, std).cuda(0)
        
        img_embs = self.vit(images).last_hidden_state
        instructions = [f"This frame is sampled at {i / fps:.1f} second." for i in indices]
        frame_embeds = self.frame_qformer(img_embs, instructions).last_hidden_state
        print(frame_embeds)

        video_token = self.video_qformer(frame_embeds)
        print(video_token)
        