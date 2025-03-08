#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

retention_ratio = 0.25 # defalt 0.25

def generate_scale_for_crop_features(base_cls_attn_scores, num_patch_width, num_patch_height, base_scale=retention_ratio, temperature=10.0):

    N = base_cls_attn_scores.shape[0]
    side_length = int(N ** 0.5)

    patch_width = max(side_length // num_patch_width, 1)
    patch_height = max(side_length // num_patch_height, 1)

    num_patches = num_patch_width * num_patch_height
    patch_scores_sum = np.zeros(num_patches)

    for idx, score in enumerate(base_cls_attn_scores):
        i, j = divmod(idx, side_length)
        
        patch_i = min(i // patch_height, num_patch_height - 1)
        patch_j = min(j // patch_width, num_patch_width - 1)
        
        patch_index = patch_i * num_patch_width + patch_j
        patch_scores_sum[patch_index] += score.item()

    # Normalize the scores and apply softmax
    shifted_scores = (patch_scores_sum - np.max(patch_scores_sum)) / temperature
    exp_scores = np.exp(shifted_scores)
    softmax_scores = exp_scores / (np.sum(exp_scores) + 1e-8)  # add a small constant to avoid division by zero

    # Calculate scales ensuring no scale exceeds 1
    if np.sum(softmax_scores) == 0:
        scales = [base_scale] * num_patches
    else:
        scales = base_scale * (1 + softmax_scores - np.mean(softmax_scores))
        scales = np.clip(scales, None, 1.0)

    return scales.tolist()


def extract_actual_dimensions(mask):
    H, W = mask.shape
    
    # Find rows and columns that contain at least one 'True' value
    rows = torch.any(mask, dim=1).nonzero(as_tuple=True)[0]
    cols = torch.any(mask, dim=0).nonzero(as_tuple=True)[0]

    if len(rows) == 0 or len(cols) == 0:
        return 0, 0

    # Calculate the actual height and width based on the bounding box of 'True' values
    actual_height = rows[-1] - rows[0] + 1
    actual_width = cols[-1] - cols[0] + 1

    return actual_width, actual_height


def get_original_indices(mask, num_img, num_tokens):

    original_indices = []

    for i in range(num_img):
        # Find indices of 'True' positions in the mask for the i-th image
        valid_indices = torch.nonzero(mask[i]).squeeze()

        # Append the valid indices for this image
        original_indices.append(valid_indices)

    return original_indices


def get_actual_crop_dimensions(mask, num_patch_width, num_patch_height):

    crop_dimensions = []

    for i in range(num_patch_height):
        for j in range(num_patch_width):
            # Extract the mask for the current patch
            patch_mask = mask[i, j, :, :]
            
            # Find non-zero indices in the patch mask
            non_zero_indices = torch.nonzero(patch_mask, as_tuple=True)
            
            if len(non_zero_indices[0]) == 0:
                # If no non-zero indices, append (0, 0) for this patch
                crop_dimensions.append((0, 0))
            else:
                # Calculate the dimensions of the crop based on non-zero indices
                min_h, max_h = non_zero_indices[0].min().item(), non_zero_indices[0].max().item() + 1
                min_w, max_w = non_zero_indices[1].min().item(), non_zero_indices[1].max().item() + 1
                
                cur_width = max_w - min_w
                cur_height = max_h - min_h
                
                crop_dimensions.append((cur_width, cur_height))

    return crop_dimensions


def interpolate_and_split_cls_attn_scores(base_cls_attn_scores, cur_width, cur_height, num_patch_width, num_patch_height, crop_dimensions):

    original_side_length = int(base_cls_attn_scores.shape[0] ** 0.5)
    base_cls_attn_scores_2d = base_cls_attn_scores.view(1, 1, original_side_length, original_side_length)

    # Interpolate the scores to the desired dimensions
    cls_attn_scores_interpolated = F.interpolate(
        base_cls_attn_scores_2d,
        size=(cur_height, cur_width), 
        mode='bilinear',               
        align_corners=False            
    ).squeeze()

    global_crops_cls_attn_scores = []

    current_h = 0 
    for i in range(num_patch_height):
        current_w = 0 
        for j in range(num_patch_width):
            index = i * num_patch_width + j
            if index >= len(crop_dimensions):
                break  

            crop_width, crop_height = crop_dimensions[index]

            # Extract and flatten the crop from interpolated scores
            crop_flattened = cls_attn_scores_interpolated[
                current_h:current_h+crop_height, 
                current_w:current_w+crop_width
            ].flatten()

            global_crops_cls_attn_scores.append(crop_flattened)

            current_w += crop_width

        if i * num_patch_width < len(crop_dimensions):
            current_h += crop_dimensions[i * num_patch_width][1]

    return global_crops_cls_attn_scores


def select_topk_crop_features(unpad_image_feature_list, unpad_cls_attn_scores_list, global_crops_cls_attn_scores, scales):

    image_feature_compress = []
    image_feature_compress_indices = []

    for i in range(len(unpad_image_feature_list)):
        valid_tokens = unpad_image_feature_list[i]
        valid_cls_scores = unpad_cls_attn_scores_list[i]
        orig_indices = torch.arange(valid_tokens.shape[0], device=valid_tokens.device)
        N, _ = valid_tokens.shape

        scale = scales[i]
        top_k = int(N * scale)

        global_crop_scores = global_crops_cls_attn_scores[i]

        norm_valid_cls_scores = F.normalize(valid_cls_scores.unsqueeze(0), p=2, dim=-1).squeeze(0)
        norm_global_crop_scores = F.normalize(global_crop_scores.unsqueeze(0), p=2, dim=-1).squeeze(0)

        combined_scores = norm_valid_cls_scores + norm_global_crop_scores
        
        _, topk_indices = torch.topk(combined_scores, top_k)

        topk_features = valid_tokens[topk_indices]
        topk_orig_indices = orig_indices[topk_indices]

        image_feature_compress.append(topk_features)
        image_feature_compress_indices.append(topk_orig_indices)

    return image_feature_compress, image_feature_compress_indices


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features, attn_maps = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_maps

    def compress_thumbnail(self, base_image_feature, base_cls_attn_scores, scale=retention_ratio):

        N, _ = base_image_feature.shape
        top_k = int(N * scale)

        # Sort indices based on attention scores in descending order
        sorted_indices = torch.argsort(base_cls_attn_scores, descending=True)
        global_compressed_indices = sorted_indices[:top_k]
        base_image_feature_compress = base_image_feature[sorted_indices[:top_k]]

        return base_image_feature_compress, global_compressed_indices

    def compress_crops(self, image_feature, 
        cls_attn_scores, base_cls_attn_scores,
        num_patch_width, num_patch_height, width, height, original_size, scales
    ):

        num_img, num_tokens, feature_dim = image_feature.shape
        original_width, original_height = original_size
        current_height, current_width = num_patch_height * height, num_patch_width * width

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        mask = torch.zeros((num_img, num_tokens), dtype=torch.bool, device=image_feature.device)
        mask = mask.view(num_patch_height, num_patch_width, height, width).permute(0, 2, 1, 3).contiguous().flatten(0, 1).flatten(1, 2)

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            mask[padding:current_height - padding, :] = True
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            mask[:, padding:current_width - padding] = True

        cur_width, cur_height = extract_actual_dimensions(mask) # Extract the actual dimensions of the valid area
        mask = mask.view(num_patch_height, height, num_patch_width, width).permute(0, 2, 1, 3).contiguous() # [num_patch_height, num_patch_width, height, width]
        
        # Get dimensions of each crop after unpadding
        crop_dimensions = get_actual_crop_dimensions(mask, num_patch_width, num_patch_height) 
        mask = mask.view(num_img, num_tokens)

        # Unpad the image features and attention scores
        unpad_image_feature_list = [image_feature[i][mask[i]] for i in range(num_img)]
        unpad_cls_attn_scores_list = [cls_attn_scores[i][mask[i]] for i in range(num_img)]
        original_indices = get_original_indices(mask, num_img, num_tokens)

        # Interpolate and split the base attention scores
        global_crops_cls_attn_scores = interpolate_and_split_cls_attn_scores(
            base_cls_attn_scores, 
            cur_width, cur_height, 
            num_patch_width, num_patch_height, 
            crop_dimensions
        )

        image_feature_compress, image_feature_compress_indices = select_topk_crop_features(
            unpad_image_feature_list, unpad_cls_attn_scores_list, 
            global_crops_cls_attn_scores, 
            scales
        )

        # Add placeholder image_newline
        final_image_features = torch.zeros((num_img, num_tokens, feature_dim), device=image_feature.device)
        valid_token_mask = torch.zeros((num_img, num_tokens), dtype=torch.bool, device=image_feature.device)

        # Update the features and mask based on compressed features and their indices
        for img_idx in range(num_img):
            feat_compressed = image_feature_compress[img_idx]
            indices_compressed = image_feature_compress_indices[img_idx]

            for feat_idx, idx in enumerate(indices_compressed):
                if idx < num_tokens:  # Ensure the index is valid
                    final_image_features[img_idx, idx] = feat_compressed[feat_idx]
                    valid_token_mask[img_idx, idx] = True

        # Reshape to the original patch dimensions and adjust the dimensions order
        final_image_features = final_image_features.view(num_patch_height, num_patch_width, height, width, -1)
        final_image_features = final_image_features.permute(4, 0, 2, 1, 3).contiguous()
        final_image_features = final_image_features.flatten(1, 2).flatten(2, 3)  # [feature_dim, num_patch_height * height, num_patch_width * width]

        valid_token_mask = valid_token_mask.view(num_patch_height, num_patch_width, height, width, -1)
        valid_token_mask = valid_token_mask.permute(4, 0, 2, 1, 3).contiguous()
        valid_token_mask = valid_token_mask.flatten(1, 2).flatten(2, 3)  # [1, num_patch_height * height, num_patch_width * width]

        # Add an image_newline token for 2D shape
        final_image_features = torch.cat((
            final_image_features, 
            self.model.image_newline[:, None, None].expand(*final_image_features.shape[:-1], 1).to(image_feature.device)
        ), dim=-1)  # [feature_dim, num_patch_height * height, num_patch_width * width + 1]
        final_image_features = final_image_features.flatten(1, 2).transpose(0, 1)

        valid_token_mask = torch.cat((
            valid_token_mask, 
            torch.ones((1, num_patch_height * height, 1), dtype=torch.bool, device=image_feature.device)
        ), dim=-1)
        valid_token_mask = valid_token_mask.flatten(1, 2).squeeze(0)

        final_image_feature_compress = final_image_features[valid_token_mask].to(dtype=torch.float16)

        return final_image_feature_compress


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features, attn_maps = self.encode_images(concat_images)
            cls_attn_scores = attn_maps[-1][:, 0, 1:]

            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:

                        # Compress global thumbnail 
                        base_image_feature = image_feature[0] 
                        base_cls_attn_scores = cls_attn_scores[0] 
                        base_image_feature_compress, global_compressed_indices = self.compress_thumbnail(base_image_feature, base_cls_attn_scores)

                        # Adaptive compression adjustment for local crops
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                            image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size) # (2, 2)

                        scales = generate_scale_for_crop_features(base_cls_attn_scores, num_patch_width, num_patch_height)

                        # Compress local crops
                        image_feature = image_feature[1:] 
                        num_crops = image_feature.shape[0]
                        cls_attn_scores = cls_attn_scores[1:] 
                        height = width = self.get_vision_tower().num_patches_per_side

                        if 'unpad' in mm_patch_merge_type:
                            original_size = image_sizes[image_idx]
                            image_feature_compress = self.compress_crops(
                                image_feature, cls_attn_scores, base_cls_attn_scores,
                                num_patch_width, num_patch_height, width, height, image_sizes[image_idx], scales
                            )
                     
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                            
                        image_feature = torch.cat((base_image_feature_compress, image_feature_compress), dim=0)
            
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
