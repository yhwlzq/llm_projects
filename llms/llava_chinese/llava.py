from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
import torch.nn as nn
import torch
from typing import Union,Optional


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):

    config_class = LlavaConfig

    def __init__(self, config:LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super().__init__(config)
        if hasattr(config,"mm_vision_tower"):
             self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]
        
        if hasattr(config,"use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self,vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower
        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        if not hasattr(self,"vision_tower"):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]

        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size//vision_config.patch_size)**2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_visioin_select_layer = mm_vision_select_layer
        
        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
             mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
             self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )
    

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None)-> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self,'vision_tower',None)

        if vision_tower is not None and (input_ids.shape[1]!=1 or self.training) and image is not None:
            vision_tower = vision_tower[0]
            with torch.no_grad():
                if type(images) is list:
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)

                else:

                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]

            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)


            dummy_image_features =  torch.zeros(256,1024,device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embbeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids==vision_tower.config.im_patch_token).sum()==0:
                    cur_input_embeds = cur_input_embeds+(0.*dummy_image_features).sum()
                    new_input_embbeds.append(cur_input_embeds)
                    cur_image_idx+=1
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cu)











