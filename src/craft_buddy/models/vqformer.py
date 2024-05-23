import torch.nn as nn
import torch
import einops
from craft_buddy.models.qformer import QFormer


class VideoQFormer(nn.Module):
    def __init__(self, max_frame_pos=32, window_size=32, window_stride=32, hidden_size=768, vision_width=768, device="cuda:0"):
        super().__init__()
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, hidden_size, device=device)
        self.qformer = QFormer(num_hidden_layers=2, cross_attention_frequency=1, device=device, vision_width=vision_width)
        self.window_size = window_size
        self.window_stride = window_stride

    def forward(self, frame_embeds):
        frame_num = frame_embeds.shape[0]
        
        position_ids = torch.arange(0, frame_num).cuda(0)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids) # (frame_num, hidden_size)
        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2) # (frame_num, query_dim - 1, hidden_size)
        
        # frame_embeds shape: (frame_num, query_num, hidden_size)
        frame_embeds = frame_embeds + frame_position_embeddings
        
        clip_hidden_state_list = []
        for i in range(0, frame_num, self.window_stride):
            clip_embeds = frame_embeds[i:i+self.window_size]
            clip_embeds = clip_embeds.unsqueeze(0) # 原本输入只是一段视频抽帧后的token，增加batch维度
            clip_embeds = einops.rearrange(clip_embeds, "b t q h -> b (t q) h")
            # clip_atts = torch.ones(1, clip_embeds.shape[1], clip_embeds.shape[1]).cuda(0)
            clip_hidden_state = self.qformer(image_embeds=clip_embeds).last_hidden_state
            clip_hidden_state_list.append(clip_hidden_state)
        
        video_hidden_state = torch.cat(clip_hidden_state_list, dim=1)
        return video_hidden_state
            