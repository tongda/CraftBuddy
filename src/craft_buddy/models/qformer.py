import torch
import torch.nn as nn
from transformers import (
    InstructBlipQFormerConfig,
    InstructBlipQFormerModel,
    BertTokenizer,
)


def _convert_blip2_state_dict(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("Qformer.bert", "qformer")
        new_key = new_key.replace("self", "attention")
        if "embeddings" in new_key:
            new_key = new_key.replace("LayerNorm", "layernorm")
        new_state_dict[new_key] = value
    return new_state_dict


class QFormer(nn.Module):
    def __init__(
        self,
        num_query_tokens=32,
        hidden_size=768,
        vision_width=1408,
        num_hidden_layers=12,
        cross_attention_frequency=2,
        tokenizer_name="bert-base-uncased",
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        with torch.device(self.device):
            self.tokenizer = BertTokenizer.from_pretrained(
                tokenizer_name, trunction_side="right"
            )
            # 原本token数量是30522，加上[DEC]后变成30523，TODO DEC是干啥用的？
            self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

            self.qformer = InstructBlipQFormerModel(
                InstructBlipQFormerConfig(
                    hidden_size=hidden_size,
                    encoder_hidden_size=vision_width,
                    hidden_dropout_prob=0.0, # 如果需要确定性结果来比对，需要关闭dropout
                    attention_probs_dropout_prob=0.0,
                    num_hidden_layers=num_hidden_layers,
                    cross_attention_frequency=cross_attention_frequency,
                )
            )
            # 因为上面增加了一个[DEC]，这里需要调用resize_token_embeddings来调整token数量
            self.qformer.resize_token_embeddings(len(self.tokenizer))

            self.query_tokens = nn.Parameter(
                torch.zeros(1, num_query_tokens, hidden_size)
            )
            # self.llm_proj = nn.Linear(hidden_size, llm_hidden_size)

    def forward(
        self,
        image_embeds,
        instructions=None,
    ):
        with torch.cuda.amp.autocast():
            if instructions is not None:
                instruction_ids = self.tokenizer(instructions, return_tensors="pt")[
                    "input_ids"
                ].to(self.device)
            else:
                instruction_ids = None
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            return self.qformer(
                instruction_ids,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
            )

    @classmethod
    def from_blip2_ckpt(cls, ckpt: str | dict):
        model = cls()
        if isinstance(ckpt, str):
            state_dict = torch.load(ckpt, map_location="cpu")["model"]
        else:
            state_dict = ckpt
        conv_state_dict = _convert_blip2_state_dict(state_dict)
        keys = model.load_state_dict(conv_state_dict, strict=False)
        print("Load from blip2 pretrained weights", keys)
        model = model.to(model.device)
        return model