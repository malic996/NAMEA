import os
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import utils as vutils

from utils.split_attention import get_att_mask

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def cnn_layerwise_gradient_scaling_hook(module, grad_in, grad_out, layer_index, total_layers,
                                        alpha=0.5, beta=2.5):
                                        # alpha=1., beta=2.0):
    """
    CNN
    """
    scale_factor = alpha + beta * (total_layers / (layer_index + 1))
    grad_in = tuple(g * scale_factor if g is not None else None for g in grad_in)
    return grad_in

def register_cnn_hooks(model, model_name, hook_handles):
    cnn_layers = list(model[1].children())
    total_layers = len(cnn_layers)
    start_idx = total_layers // 3  # 从中间层开始

    for i, layer in enumerate(cnn_layers[start_idx:], start=start_idx):
    # for i, layer in enumerate(cnn_layers):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            handle = layer.register_full_backward_hook(
                lambda mod, grad_in, grad_out: cnn_layerwise_gradient_scaling_hook(
                    mod, grad_in, grad_out, i, total_layers
                )
            )
            hook_handles.append(handle)  # save

def register_vit_hooks(model, model_name, hook_handles):
    """
    ViT
    """
    u = 1.0
    def attn_tgr(module, grad_in, grad_out, gamma):
        mask = torch.ones_like(grad_in[0]) * gamma
        out_grad = mask * grad_in[0][:]
        return (out_grad,)

    def q_tgr(module, grad_in, grad_out, gamma):
        # cait Q only uses class token
        mask = torch.ones_like(grad_in[0]) * gamma
        out_grad = mask * grad_in[0][:]
        return (out_grad, grad_in[1], grad_in[2])


    def v_tgr(module, grad_in, grad_out, name):
        """
        ViT(DeiT-Tiny, ViT-Tiny）
        """
        # print(f"[DEBUG] grad_in[0].shape: {grad_in[0].shape}")  # Debug
        B = 20
        if grad_in[0] is None:
            return grad_in

        if grad_in[0].dim() == 3:
            B_, N, D = grad_in[0].shape
            B_actual = B_  # or check if B_ == B
            grad_reshaped = grad_in[0]
        elif grad_in[0].dim() == 2 and B is not None:
            BN, D = grad_in[0].shape
            if BN % B != 0:
                print("[ERROR] (B*N, D) can not restore (B, N, D)")
                return grad_in
            N = BN // B
            B_actual = B
            grad_reshaped = grad_in[0].view(B, N, D)
        else:
            print("[WARNING] shape not matched for token-wise scaling:", grad_in[0].shape)
            return grad_in

            t_mus = torch.mean(torch.abs(grad_reshaped), dim=[0, 1])  # shape = (B, D)
            print(f"[DEBUG] t_mus.shape: {t_mus.shape}")  # Debug
            t_mus_np = t_mus.cpu().numpy()

            mu = np.mean(t_mus_np)
            print(f"[DEBUG] t_mus.shape: {mu.shape}")  # Debug
            std = np.std(t_mus_np)
            muustd = 1.0 * mu - 0.5 * std

            t_factor_bool = t_mus_np < muustd
            t_temp = np.tanh(abs((t_mus_np - mu) / (std + 1e-6)))
            t_factor = np.array(t_factor_bool).astype(np.float32)
            t_factor[t_factor == False] = t_temp[t_factor == False]

            t_factor_tensor = torch.from_numpy(t_factor).to(grad_reshaped.device)  # shape=(B, N)

            # grad_scaled = grad_reshaped * t_factor_tensor.view(B_actual, 1, D)
            grad_scaled = (grad_reshaped[:, :, :] * torch.from_numpy(t_factor).
                                   to(grad_in[0].device).view(1, 1, D))

            if grad.dim() == 2 and B is not None:
                grad_final = grad_scaled.view(B_actual * N, D)
            else:
                grad_final = grad_scaled
            mask = torch.ones_like(grad_in[0])
            grad_final = grad_final * mask

            return (grad_final,)


    def mlp_tgr(module, grad_in, grad_out, gamma):
        mask = torch.ones_like(grad_in[0]) * gamma
        out_grad = mask * grad_in[0][:]
        for i in range(len(grad_in)):
            if i == 0:
                return_dics = (out_grad,)
            else:
                return_dics = return_dics + (grad_in[i],)
        return return_dics

    attn_tgr_hook = partial(attn_tgr, gamma=0.5)
    v_tgr_hook = v_tgr
    mlp_tgr_hook = partial(mlp_tgr, gamma=1.0)

    for i in range(12):
        hook_handles.append(model[1].blocks[i].attn.attn_drop.register_full_backward_hook(
            attn_tgr_hook))

        hook_handles.append(model[1].blocks[i].attn.qkv.register_full_backward_hook(
            partial(v_tgr_hook, name=f"model.blocks[{i}].attn.qkv")))
        hook_handles.append(model[1].blocks[i].mlp.register_full_backward_hook(mlp_tgr_hook))




# ========================== release Hook ========================== #
def remove_hooks(hook_handles):
    """
    remove register Hook
    """
    for handle in hook_handles:
        handle.remove()



def MI_FGSM_SMER_meta(surrogate_models,images, labels, args, num_iter=10):
    hook_handles = []
    eps = args.eps/255.0
    alpha = args.alpha/255.0
    beta = alpha
    # you can adjust the threshold for the best performance
    # for example
    threshold_list_cnn = [0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.65, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85, 0.85]
    momentum = args.momentum
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    clean_img = images.clone().detach()
    m = len(surrogate_models)
    m_smer = m * 4
    grad_final = 0
    grad_noatt = 0
    x_local = images.clone().detach()
    for i in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        if x_local.grad is not None:
            x_local.grad.zero_()
        images = Variable(images, requires_grad=True)
        x_local = Variable(x_local, requires_grad=True)
        x_inner = images.detach()
        x_inner_noatt = x_local.detach()
        # x_inner_noatt = images.detach()
        x_before = images.clone()
        x_inner_noatt_before = x_local.clone()
        noise_inner_global = torch.zeros([m_smer, *images.shape]).to(images.device)
        noise_inner_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        noise_inner_noatt_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        mask = torch.zeros([m_smer, *images.shape]).to(images.device)
        mask_noatt = torch.zeros([m_smer, *images.shape]).to(images.device)
        grad_inner1 = torch.zeros_like(images)
        grad_inner2 = torch.zeros_like(images)
        grad_inner_before = torch.zeros_like(images)

        options = []
        for i in range(int(m_smer / m)):
            options_single=[j for j in range(m)]
            np.random.shuffle(options_single)
            options.append(options_single)
        options = np.reshape(options,-1)
        # meta train
        adv_list_pre = []
        grad_list_pre = []

        for j in range(m_smer):
            option = options[j]
            grad_single = surrogate_models[option]
            adv_list_pre.append(x_inner.clone())
            x_inner.requires_grad = True
            out = grad_single((x_inner))
            loss1 = F.cross_entropy(out, labels)
            noise_im_inner = torch.autograd.grad(loss1, x_inner, retain_graph=True)[0]
            noise_im_inner = noise_im_inner.clone()
            x_inner.requires_grad = False
            noise_inner1 = noise_im_inner
            noise_inner1 = noise_inner1 / torch.mean(torch.abs(noise_inner1), dim=[1, 2, 3], keepdims=True)
            grad_inner1 = momentum * grad_inner1 + noise_inner1

            x_inner = x_inner + beta * torch.sign(grad_inner1)
            x_inner = clip_by_tensor(x_inner, image_min, image_max)
            grad_list_pre.append(grad_inner1.clone())
            noise_inner_all[j] = grad_inner1.clone()
        # mate test
        for j in range(m_smer):
            remove_hooks(hook_handles)
            option = options[j]
            grad_single = surrogate_models[option]

            if option == 0:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[j],
                                       model_name='res18')
                register_cnn_hooks(grad_single, "ResNet-18", hook_handles)
            elif option == 1:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[j],
                                       model_name='incv3')
                register_cnn_hooks(grad_single, "Inception-V3", hook_handles)
            elif option == 2:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[j],
                                       model_name='vit')
                register_vit_hooks(grad_single, "ViT-Tiny", hook_handles)
            else:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[j],
                                       model_name='deit')
                register_vit_hooks(grad_single, "DeiT-Tiny", hook_handles)
            no_att_raw = no_att1
            no_att1 = 1 - no_att1
            noise_padding = torch.randn_like(x_inner_noatt)
            noise_padding = torch.clip(noise_padding * no_att_raw, 0., 1.)
            x_inner_noatt = (x_inner_noatt) * no_att1
            x_inner_noatt = (x_inner_noatt) + noise_padding
            x_inner_noatt = torch.clamp(x_inner_noatt, 0., 1.)

            x_inner_noatt.requires_grad = True

            out_noatt = grad_single((x_inner_noatt))
            loss2 = F.cross_entropy(out_noatt, labels)
            noise_im_inner2 = torch.autograd.grad(loss2, x_inner_noatt, retain_graph=True)[0]
            noise_im_inner2 = noise_im_inner2.clone()
            if x_inner_noatt.grad is not None:
                x_inner_noatt.grad.zero_()
            x_inner_noatt.requires_grad = False
            noise_inner2 = noise_im_inner2
            noise_inner2 = noise_inner2 / torch.mean(torch.abs(noise_inner2), dim=[1, 2, 3], keepdims=True)
            # grad_inner = grad_inner + noise_inner
            grad_inner2 = 1. * grad_inner2 + noise_inner2

            grad_inner = 1.0 * grad_list_pre[j] + 1. * grad_inner2 * no_att1
            grad_inner = grad_inner / torch.mean(torch.abs(grad_inner), dim=[1, 2, 3], keepdims=True)
            grad_inner = grad_inner + 1. * grad_inner_before
            grad_inner_before = grad_inner

            x_inner_noatt = x_inner_noatt + beta * torch.sign(grad_inner2)
            x_inner_noatt = clip_by_tensor(x_inner_noatt, image_min, image_max)
            noise_inner_global[j] = grad_inner_before.clone()
            noise_inner_noatt_all[j] = grad_inner2.clone()
            mask[j] = no_att1.clone()
            mask_noatt[j] = no_att_raw.clone()
        mask_final = mask[-1].clone()
        noise1 = noise_inner_all[-1].clone()
        noise2 = noise_inner_noatt_all[-1].clone()
        noise1 = noise1 / torch.mean(torch.abs(noise1), dim=[1, 2, 3], keepdims=True)
        noise2 = noise2 / torch.mean(torch.abs(noise2), dim=[1, 2, 3], keepdims=True)
        noise_global = noise1 + noise2 * mask_final

        noise_global = noise_global / torch.mean(torch.abs(noise_global), dim=[1, 2, 3], keepdims=True)
        grad_final = noise_global + grad_final

        # grad_final = noise_global
        images = x_before + alpha * torch.sign(grad_final)
        x_local = x_inner_noatt_before + alpha * torch.sign(grad_final)
        images = clip_by_tensor(images, image_min, image_max)
        x_local = clip_by_tensor(x_local, image_min, image_max)
    return images




def I_FGSM_SMER_meta(surrogate_models,images, labels, args, num_iter=10):
    hook_handles = []  # 存储 Hook 句柄
    eps = args.eps/255.0
    alpha = args.alpha/255.0
    beta = alpha
    # you can adjust the threshold for the best performance
    # for example
    threshold_list_cnn = [0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.65, 0.65, 0.65, 0.65, 0.85, 0.85, 0.85, 0.85]
    momentum = args.momentum
    image_min = clip_by_tensor(images - eps, 0.0, 1.0)
    image_max = clip_by_tensor(images + eps, 0.0, 1.0)
    clean_img = images.clone().detach()
    m = len(surrogate_models)
    m_smer = m * 4
    grad_final = 0
    grad_noatt = 0
    x_local = images.clone().detach()
    for i in range(num_iter):
        if images.grad is not None:
            images.grad.zero_()
        if x_local.grad is not None:
            x_local.grad.zero_()
        images = Variable(images, requires_grad=True)
        x_local = Variable(x_local, requires_grad=True)
        x_inner = images.detach()
        # x_inner_noatt = x_local.detach()
        x_inner_noatt = images.detach()
        x_before = images.clone()
        x_inner_noatt_before = x_local.clone()
        noise_inner_global = torch.zeros([m_smer, *images.shape]).to(images.device)
        noise_inner_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        noise_inner_noatt_all = torch.zeros([m_smer, *images.shape]).to(images.device)
        mask = torch.zeros([m_smer, *images.shape]).to(images.device)
        mask_noatt = torch.zeros([m_smer, *images.shape]).to(images.device)
        grad_inner1 = torch.zeros_like(images)
        grad_inner2 = torch.zeros_like(images)
        grad_inner_before = torch.zeros_like(images)

        options = []
        for i in range(int(m_smer / m)):
            options_single=[j for j in range(m)]
            np.random.shuffle(options_single)
            options.append(options_single)
        options = np.reshape(options,-1)
        # meta train
        adv_list_pre = []
        grad_list_pre = []

        for j in range(m_smer):
            option = options[j]
            grad_single = surrogate_models[option]
            adv_list_pre.append(x_inner.clone())
            x_inner.requires_grad = True
            out = grad_single((x_inner))
            loss1 = F.cross_entropy(out, labels)
            noise_im_inner = torch.autograd.grad(loss1, x_inner, retain_graph=True)[0]
            noise_im_inner = noise_im_inner.clone()
            x_inner.requires_grad = False
            noise_inner1 = noise_im_inner
            noise_inner1 = noise_inner1 / torch.mean(torch.abs(noise_inner1), dim=[1, 2, 3], keepdims=True)
            grad_inner1 = 1. * grad_inner1 + noise_inner1
            x_inner = x_inner + beta * torch.sign(grad_inner1)
            x_inner = clip_by_tensor(x_inner, image_min, image_max)
            grad_list_pre.append(grad_inner1.clone())
            noise_inner_all[j] = grad_inner1.clone()

        # mate test
        for k in range(m_smer):
            remove_hooks(hook_handles)
            option = options[k]
            grad_single = surrogate_models[option]
            if option == 0:
                no_att1 = get_att_mask((adv_list_pre[k]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[k],
                                       model_name='res18')
                register_cnn_hooks(grad_single, "ResNet-18", hook_handles)
            elif option == 1:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[k],
                                       model_name='incv3')
                register_cnn_hooks(grad_single, "Inception-V3", hook_handles)

            elif option == 2:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[k],
                                       model_name='vit')
                register_vit_hooks(grad_single, "ViT-Tiny", hook_handles)
            else:
                no_att1 = get_att_mask((adv_list_pre[j]), labels,
                                       model=grad_single, threshold=threshold_list_cnn[k],
                                       model_name='deit')
                register_vit_hooks(grad_single, "DeiT-Tiny", hook_handles)

            no_att_raw = no_att1
            no_att1 = 1 - no_att1
            noise_padding = torch.randn_like(x_inner_noatt)
            noise_padding = torch.clip(noise_padding * no_att_raw, 0., 1.)

            x_inner_noatt = (x_inner_noatt) * no_att1
            x_inner_noatt = (x_inner_noatt) + noise_padding
            x_inner_noatt = torch.clamp(x_inner_noatt, 0., 1.)

            x_inner_noatt.requires_grad = True

            out_noatt = grad_single((x_inner_noatt))
            loss2 = F.cross_entropy(out_noatt, labels)
            noise_im_inner2 = torch.autograd.grad(loss2, x_inner_noatt, retain_graph=True)[0]
            # noise_im_inner2 = torch.autograd.grad(loss2, x_inner, retain_graph=True)[0]
            noise_im_inner2 = noise_im_inner2.clone()
            if x_inner_noatt.grad is not None:
                x_inner_noatt.grad.zero_()
            x_inner_noatt.requires_grad = False
            noise_inner2 = noise_im_inner2
            noise_inner2 = noise_inner2 / torch.mean(torch.abs(noise_inner2), dim=[1, 2, 3], keepdims=True)
            # grad_inner = grad_inner + noise_inner
            grad_inner2 = 1. * grad_inner2 + noise_inner2

            grad_inner = 1.0 * grad_list_pre[k] * no_att_raw + 1. * grad_inner2 * no_att1
            grad_inner_before = grad_inner

            x_inner_noatt = x_inner_noatt + beta * torch.sign(grad_inner2)
            x_inner_noatt = clip_by_tensor(x_inner_noatt, image_min, image_max)


            noise_inner_global[k] = grad_inner_before.clone()
            noise_inner_noatt_all[k] = grad_inner2.clone()
            mask[k] = no_att1.clone()
            mask_noatt[k] = no_att_raw.clone()
        noise_global = noise_inner_global[-1].clone()
        grad_final = noise_global

        images = x_before + alpha * torch.sign(grad_final)
        x_local = x_inner_noatt_before + alpha * torch.sign(grad_final)
        images = clip_by_tensor(images, image_min, image_max)
        x_local = clip_by_tensor(x_local, image_min, image_max)
    return images