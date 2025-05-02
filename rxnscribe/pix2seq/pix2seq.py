# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq model and criterion classes.
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from torch import nn

from .misc import nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer
from transformers import GenerationConfig

import numpy as np


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer, use_hf = False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_bins: number of bins for each side of the input image
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = 256 if use_hf else transformer.d_model
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone

        self.use_hf = use_hf

        

    def forward(self, image_tensor, targets=None, max_len=500, cheat = None):
        """ 
        image_tensor:
        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all vocabulary.
                            Shape= [batch_size, num_sequence, num_vocal]
        """

        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)
        features, pos = self.backbone(image_tensor)  
        #print(len(features))
        #print(pos.size()) 
        '''
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack = True, record_shapes=True) as prof:
                with record_function("model_inference"):
                    features, pos = self.backbone(image_tensor)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        prof.export_stacks("/tmp/profiler_stacks_cuda_A6000_16_backbone.txt", "self_cuda_time_total") 
        '''
        src, mask = features[-1].decompose()
        assert mask is not None
        mask = torch.zeros_like(mask).bool()

        src = self.input_proj(src)

        
        

        if self.use_hf: 
            if targets is not None:
                '''
                logits = self.transformer(src)



                



                input_seq, input_len = targets

                logits = logits.reshape(-1, 2094)

                loss = self.loss_fn(logits, input_seq.view(-1))

                return loss, loss
                '''


                '''
                output_logits = self.transformer(src, input_seq[:, 1:], mask, pos[-1])
                return output_logits[:, :-1]
                '''
                #print(input_seq)
                input_seq, input_len = targets
                input_seq = input_seq[:, 1:]
                bs = src.shape[0]
                src = src.flatten(2).permute(0, 2, 1)
                #b x c x h x w to b x hw x c
                pos_embed = pos[-1].flatten(2).permute(0, 2, 1)
                max_len = input_seq.size(1)
                indices = torch.arange(max_len).unsqueeze(0).expand_as(input_seq).to(src.device)
                mask = indices >= input_len - torch.ones(input_len.shape).to(src.device)
                masked_input_seq = input_seq.masked_fill(mask, -100)
                #print("input_seq "+str(input_seq))
                #print("masked_input "+str(masked_input_seq))
                #src = src + pos_embed #unclear if this line is needed...
                '''
                decoder_input = torch.cat(
                    [
                        nn.Embedding(1, 256).to(src.device).weight.unsqueeze(0).repeat(bs, 1, 1),
                        nn.Embedding(2092, 256).to(src.device)(input_seq)
                    ], dim = 1
                )
                '''
                #decoder_mask = torch.full(decoder_input.shape[:2], False, dtype = torch.bool).to(src.device)
                #decoder_mask[:, 0] = True
                output = self.transformer(inputs_embeds = src,labels = masked_input_seq)
                #print("output logits " + str(torch.argmax(output["logits"], dim = 2)) + "target labels "+ str(masked_input_seq))

                #print(output["logits"].shape)

                return output["logits"], output["loss"] 
            else:
                '''
                logits = self.transformer(src)

                print(logits.shape)

                return self.transformer(src).argmax(dim = 1), self.transformer(src).argmax(dim = 1)
                '''

                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack = True, record_shapes=True) as prof:
                #    with record_function("model_inference"):
                #print(pos[-1])
                #output_seqs, output_scores = self.transformer(src, None, mask, pos[-1], max_len=max_len)
                '''
                flatten src from B x C x H x W into B x HW x C and pass in as input_embeds
                potentially flatten pos[-1] as well and add to input embeds
                '''
                bs = src.shape[0]
                src = src.flatten(2).permute(0, 2, 1)
                generation_config = GenerationConfig(max_new_tokens = max_len, bos_token_id = 2002, eos_token_id = 2092, pad_token_id = 2001, output_hidden_states = True)
                #output = self.transformer.generate(inputs_embeds = src, generation_config = generation_config, return_dict_in_generate=True, output_scores=True)
                #transition_scores = self.transformer.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
                #for tok, score in zip(output.sequences[0], transition_scores[0]):
                #    print(f"| {tok:5d} | {score.to('cpu').numpy():.3f} | {np.exp(score.to('cpu').numpy()):.2%}")
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                #prof.export_stacks("/tmp/profiler_stacks_cpu_A6000_16_decoder.txt", "self_cpu_time_total")
                #print("loss "+str(output.loss))
                
                #encoder_outputs = self.transformer.encoder(inputs_embeds = src)
                '''
                print(cheat)
                print("own predictions")
                print(cheat['coref'][0][:, :3])
                print(self.transformer.decoder(input_ids = cheat['coref'][0][:, :3].to(src.device), encoder_hidden_states = torch.rand_like(encoder_outputs[0]).to(src.device)).logits.argmax(dim = 2))
                print(self.transformer.decoder(input_ids = cheat['coref'][0][:, :4].to(src.device), encoder_hidden_states = torch.rand_like(encoder_outputs[0]).to(src.device)).logits.argmax(dim = 2))
                print(self.transformer.decoder(input_ids = cheat['coref'][0][:, :5].to(src.device), encoder_hidden_states = torch.rand_like(encoder_outputs[0]).to(src.device)).logits.argmax(dim = 2))
                '''

                #input_seq, input_len = cheat['bbox']
                #input_seq = input_seq[:, 1:]
                #b x c x h x w to b x hw x c
                #max_len = input_seq.size(1)
                #indices = torch.arange(max_len).unsqueeze(0).expand_as(input_seq).to(src.device)
                #mask = indices >= input_len - torch.ones(input_len.shape).to(src.device)
                #masked_input_seq = input_seq.masked_fill(mask, -100)
                #output = self.transformer(inputs_embeds = src,labels = masked_input_seq) 
                #print("output logits " + str(torch.argmax(output["logits"], dim = 2)) + "target labels "+ str(masked_input_seq))
                outputs = self.transformer.generate(inputs_embeds = src, generation_config = generation_config)
            
                return outputs, outputs
        else:
            if targets is not None:
                input_seq, input_len = targets
                output_logits = self.transformer(src, input_seq[:, 1:], mask, pos[-1])
                return output_logits[:, :-1]
            else:
                output_seqs, output_scores = self.transformer(src, None, mask, pos[-1], max_len=max_len)
                return output_seqs, output_scores



def build_pix2seq_model(args, tokenizer):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223


    

    backbone = build_backbone(args)
    transformer = build_transformer(args, tokenizer)

    model = Pix2Seq(backbone, transformer, use_hf = args.use_hf_transformer)

    if args.pix2seq_ckpt is not None:
        checkpoint = torch.load(args.pix2seq_ckpt, map_location='cpu')
        if args.use_hf_transformer:
            new_dict = {}
            #print(checkpoint['state_dict'].keys())
            for key in checkpoint['state_dict']:
                new_dict[key[6:]] = checkpoint['state_dict'][key]
            model.load_state_dict(new_dict, strict = False)
        else:
            model.load_state_dict(checkpoint['model'])

    return model
