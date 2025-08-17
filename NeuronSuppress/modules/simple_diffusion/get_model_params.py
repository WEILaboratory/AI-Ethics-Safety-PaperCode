import torch

from torch import nn
from modules.encoder import CLIPEncoder

from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel

pretrain_model_path = '/home/junlei/.cache/huggingface/lansinuote/diffsion_from_scratch.params'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_params = CLIPTextModel.from_pretrained(pretrain_model_path, subfolder='text_encoder')
VAE_params = AutoencoderKL.from_pretrained(pretrain_model_path, subfolder='vae')
Unet_params = UNet2DConditionModel.from_pretrained(pretrain_model_path, subfolder='unet')

def print_model_architecture(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
# print("Loaded UNet2DConditionModel model architecture:")
# print_model_architecture(Unet_params)


def init_CLIP_encoder_params(encoder):

    encoder.text_embed.embed.load_state_dict(CLIP_params.text_model.embeddings.token_embedding.state_dict())
    encoder.text_embed.pos_embed.load_state_dict(CLIP_params.text_model.embeddings.position_embedding.state_dict())

    for encoder_layer, layer_params in zip(encoder[1:], CLIP_params.text_model.encoder.layers):
        if isinstance(encoder_layer, CLIPEncoder):
            #  first norm
            encoder_layer.s1[0].load_state_dict(layer_params.layer_norm1.state_dict())

            #  attention q, k, v and out
            for name, proj in zip(['q', 'k', 'v', 'out'], [layer_params.self_attn.q_proj, layer_params.self_attn.k_proj, layer_params.self_attn.v_proj, layer_params.self_attn.out_proj]):
                getattr(encoder_layer.s1[1], name).load_state_dict(proj.state_dict())

            #  norm
            encoder_layer.s2[0].load_state_dict(layer_params.layer_norm2.state_dict())

            #  MLP fc1
            encoder_layer.s2[1].load_state_dict(layer_params.mlp.fc1.state_dict())

            #  MLP fc2
            encoder_layer.s3.load_state_dict(layer_params.mlp.fc2.state_dict())
        else:
            break
    encoder.LayerNorm.load_state_dict(CLIP_params.text_model.final_layer_norm.state_dict())

    print("CLIP encoder loading successfully!")
    # encoder.eval()
    # CLIP_params.eval()

    # a = encoder(torch.arange(77).unsqueeze(dim=0))
    # b = CLIP_params(torch.arange(77).unsqueeze(dim=0)).last_hidden_state
    # print(a.shape, b.shape)
    # print((a == b).all())

def load_resnet(model, params):
    model.ResnetModel.groupnorm1.load_state_dict(params.norm1.state_dict())
    model.ResnetModel.conv1.load_state_dict(params.conv1.state_dict())
    model.ResnetModel.groupnorm2.load_state_dict(params.norm2.state_dict())
    model.ResnetModel.conv2.load_state_dict(params.conv2.state_dict())

    if isinstance(model.residual, nn.Module):    
        model.residual.load_state_dict(params.conv_shortcut.state_dict())

def load_attention(model, params):
    model.norm.load_state_dict(params.group_norm.state_dict())
    model.q.load_state_dict(params.query.state_dict())
    model.k.load_state_dict(params.key.state_dict())
    model.v.load_state_dict(params.value.state_dict())
    model.out.load_state_dict(params.proj_attn.state_dict())


def init_vae_params(vae):
    # print(VAE_params.encoder.conv_in.state_dict().items())

    #  encoder.in
    vae.encoder.conv1.load_state_dict(VAE_params.encoder.conv_in.state_dict())
    print()

    for i in range(4):
        load_resnet(vae.encoder[i + 1][0], VAE_params.encoder.down_blocks[i].resnets[0])
        load_resnet(vae.encoder[i + 1][1], VAE_params.encoder.down_blocks[i].resnets[1])

        if i != 3:
            vae.encoder[i + 1][2][1].load_state_dict(VAE_params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())
    
    #  load mid_encoder_block
    load_resnet(vae.encoder.mid_encoder_block[0], VAE_params.encoder.mid_block.resnets[0])
    load_attention(vae.encoder.mid_encoder_block[1], VAE_params.encoder.mid_block.attentions[0])
    load_resnet(vae.encoder.mid_encoder_block[2], VAE_params.encoder.mid_block.resnets[1])

    #  load out_encoder_block
    vae.encoder.out_encoder_block[0].load_state_dict(VAE_params.encoder.conv_norm_out.state_dict())
    vae.encoder.out_encoder_block[2].load_state_dict(VAE_params.encoder.conv_out.state_dict())

    #  load encoder norm_layer
    vae.encoder.norm_layer.load_state_dict(VAE_params.quant_conv.state_dict())
    
    ###-----------------------------------------------------------------------------###
    #  load decoder norm_layer
    vae.decoder.norm_layer.load_state_dict(VAE_params.post_quant_conv.state_dict())
    #  load decoder in block
    vae.decoder.conv1.load_state_dict(VAE_params.decoder.conv_in.state_dict())
    #  load decoder mid block
    load_resnet(vae.decoder.mid_decoder_block[0], VAE_params.decoder.mid_block.resnets[0])
    load_attention(vae.decoder.mid_decoder_block[1], VAE_params.decoder.mid_block.attentions[0])
    load_resnet(vae.decoder.mid_decoder_block[2], VAE_params.decoder.mid_block.resnets[1])
    #  load decoder up block
    for i in range(4):
        load_resnet(vae.decoder[i + 3][0], VAE_params.decoder.up_blocks[i].resnets[0])
        load_resnet(vae.decoder[i + 3][1], VAE_params.decoder.up_blocks[i].resnets[1])
        load_resnet(vae.decoder[i + 3][2], VAE_params.decoder.up_blocks[i].resnets[2])

        if i != 3:
            vae.decoder[i + 3][4].load_state_dict(VAE_params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())
    #  load decoder out block
    vae.decoder.out_decoder_block[0].load_state_dict(VAE_params.decoder.conv_norm_out.state_dict())
    vae.decoder.out_decoder_block[2].load_state_dict(VAE_params.decoder.conv_out.state_dict())

    print("VAE loading successfully!")
    # en_data = torch.randn(1, 3, 512, 512)
    # en_a = vae.encoder(en_data)
    # en_b = VAE_params.encode(en_data).latent_dist.parameters
    # print((en_a == en_b).all())

def load_transformer(model, param):

    #  load input block
    model.norm_in.load_state_dict(param.norm.state_dict())
    model.cnn_in.load_state_dict(param.proj_in.state_dict())

    #  load attention_block
    model.norm_atten0.load_state_dict(param.transformer_blocks[0].norm1.state_dict())
    model.norm_atten1.load_state_dict(param.transformer_blocks[0].norm2.state_dict())

    for (name, proj) in zip(['q', 'k', 'v', 'out'], [param.transformer_blocks[0].attn1.to_q, param.transformer_blocks[0].attn1.to_k, param.transformer_blocks[0].attn1.to_v, param.transformer_blocks[0].attn1.to_out[0]]):
        getattr(model.cross_atten1, name).load_state_dict(proj.state_dict())
        # print(proj.state_dict(), "#######\n")

    for (name, proj) in zip(['q', 'k', 'v', 'out'], [param.transformer_blocks[0].attn2.to_q, param.transformer_blocks[0].attn2.to_k, param.transformer_blocks[0].attn2.to_v, param.transformer_blocks[0].attn2.to_out[0]]):
        getattr(model.cross_atten2, name).load_state_dict(proj.state_dict())
        # print(proj.state_dict(), "#######\n")
    
    # return
    #  load activate block
    model.norm_act.load_state_dict(param.transformer_blocks[0].norm3.state_dict())
    model.fc0.load_state_dict(param.transformer_blocks[0].ff.net[0].proj.state_dict())
    model.fc1.load_state_dict(param.transformer_blocks[0].ff.net[2].state_dict())

    #  load output block
    model.cnn_out.load_state_dict(param.proj_out.state_dict())


def load_unet_resnet(model, params):
    model.time_embedding[1].load_state_dict(params.time_emb_proj.state_dict())

    model.conv_block1[0].load_state_dict(params.norm1.state_dict())
    model.conv_block1[2].load_state_dict(params.conv1.state_dict())

    model.conv_block2[0].load_state_dict(params.norm2.state_dict())
    model.conv_block2[2].load_state_dict(params.conv2.state_dict())
    
    if isinstance(model.residual, torch.nn.Module):
        model.residual.load_state_dict(params.conv_shortcut.state_dict())

def load_unet_down_block(model, params):
    load_transformer(model.transformer1, params.attentions[0])
    load_transformer(model.transformer2, params.attentions[1])

    load_unet_resnet(model.resnet1, params.resnets[0])
    load_unet_resnet(model.resnet2, params.resnets[1])

    model.downsample.load_state_dict(params.downsamplers[0].conv.state_dict())


def load_unet_up_block(model, params):
    load_unet_resnet(model.resnet1, params.resnets[0])
    load_unet_resnet(model.resnet2, params.resnets[1])
    load_unet_resnet(model.resnet3, params.resnets[2])

    load_transformer(model.transformer1, params.attentions[0])
    load_transformer(model.transformer2, params.attentions[1])
    load_transformer(model.transformer3, params.attentions[2])

    if isinstance(model.upsample, torch.nn.Module):
        model.upsample.conv.load_state_dict(params.upsamplers[0].conv.state_dict())

def init_unet_params(unet):

    unet.in_vae.load_state_dict(Unet_params.conv_in.state_dict())
    unet.in_time[0].load_state_dict(Unet_params.time_embedding.linear_1.state_dict())
    unet.in_time[2].load_state_dict(Unet_params.time_embedding.linear_2.state_dict())

    #  load down block layer
    load_unet_down_block(unet.down_block1, Unet_params.down_blocks[0])
    load_unet_down_block(unet.down_block2, Unet_params.down_blocks[1])
    load_unet_down_block(unet.down_block3, Unet_params.down_blocks[2])

    load_unet_resnet(unet.down_resnet1, Unet_params.down_blocks[3].resnets[0])
    load_unet_resnet(unet.down_resnet2, Unet_params.down_blocks[3].resnets[1])

    #  load mid block layer
    load_transformer(unet.mid_transformer, Unet_params.mid_block.attentions[0])
    load_unet_resnet(unet.mid_resnet1, Unet_params.mid_block.resnets[0])
    load_unet_resnet(unet.mid_resnet2, Unet_params.mid_block.resnets[1])

    #  load up block layer
    load_unet_resnet(unet.up_resnet1, Unet_params.up_blocks[0].resnets[0])
    load_unet_resnet(unet.up_resnet2, Unet_params.up_blocks[0].resnets[1])
    load_unet_resnet(unet.up_resnet3, Unet_params.up_blocks[0].resnets[2])
    unet.up_in[1].load_state_dict(Unet_params.up_blocks[0].upsamplers[0].conv.state_dict())

    load_unet_up_block(unet.up_block1, Unet_params.up_blocks[1])
    load_unet_up_block(unet.up_block2, Unet_params.up_blocks[2])
    load_unet_up_block(unet.up_block3, Unet_params.up_blocks[3])

    #  load output layer
    unet.out[0].load_state_dict(Unet_params.conv_norm_out.state_dict())
    unet.out[2].load_state_dict(Unet_params.conv_out.state_dict())

    print("Unet loading successfully!")

    # out_vae = torch.randn(1, 4, 64, 64)
    # out_encoder = torch.randn(1, 77, 768)
    # time = torch.LongTensor([26])

    # a = unet(out_vae=out_vae, out_encoder=out_encoder, time=time)
    # b = Unet_params(out_vae, time, out_encoder).sample

    # print("Unet", (a == b).all())
