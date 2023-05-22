#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
streamlit app demo
how to run:
streamlit run app.py --server.port 8501

@author: Tu Bui @surrey.ac.uk
"""
import os, sys, torch 
import argparse
from pathlib import Path
import numpy as np
import pickle
import pytorch_lightning as pl
from torchvision import transforms
import argparse
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from tools.augment_imagenetc import RandomImagenetC
from io import BytesIO
from tools.helpers import welcome_message
from tools.ecc import BCH, RSC

import streamlit as st

# noise = RandomImagenetC(phase='test')
# corrupt_methods = [noise.method_names[i] for i in noise.corrupt_ids]
model_names = ['RoSteALS']
SECRET_LEN = 100

def unormalize(x):
    # convert x in range [-1, 1], (B,C,H,W), tensor to [0, 255], uint8, numpy, (B,H,W,C)
    x = torch.clamp((x + 1) * 127.5, 0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return x

def to_bytes(x, mime):
    x = Image.fromarray(x)
    buf = BytesIO()
    f = "JPEG" if mime == 'image/jpeg' else "PNG"
    x.save(buf, format=f)
    byte_im = buf.getvalue()
    return byte_im

def load_RoSteALS():
    # prioritise secret recovery
    config_file = 'models/VQ4_mir_inference.yaml'
    weight_file = 'models/epoch=000017-step=000449999.ckpt'
    config = OmegaConf.load(config_file).model
    secret_len = config.params.control_config.params.secret_len
    assert SECRET_LEN == secret_len
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)
    state_dict = torch.load(weight_file, map_location=torch.device('cpu'))
    if 'global_step' in state_dict:
        print(f'Global step: {state_dict["global_step"]}, epoch: {state_dict["epoch"]}')

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print(f'Missed keys: {misses}\nIgnore keys: {ignores}')
    model.eval()
    return model

def embed_secret(model_name, model, cover, tform, secret):
    if model_name == 'RoSteALS':
        w, h = cover.size
        with torch.no_grad():
            im = tform(cover).unsqueeze(0)  # 1, 3, 256, 256
            z = model.encode_first_stage(im)
            z_embed, _ = model(z, None, secret)
            stego = model.decode_first_stage(z_embed)  # 1, 3, 256, 256
            res = (stego.clamp(-1,1) - im)  # (1,3,256,256) residual
            res = torch.nn.functional.interpolate(res, (h,w), mode='bilinear')
            res = res.permute(0,2,3,1).cpu().numpy()  # (1,256,256,3)
            stego_uint8 = np.clip(res[0] + np.array(cover)/127.5-1., -1,1)*127.5+127.5  # (256, 256, 3), ndarray, uint8
            stego_uint8 = stego_uint8.astype(np.uint8)
    else:
        raise NotImplementedError
    return stego_uint8


def decode_secret(model_name, model, im, tform):
    if model_name == 'RoSteALS':
        with torch.no_grad():
            im = tform(im).unsqueeze(0)  # 1, 3, 256, 256
            secret_pred = (model.decoder(im) > 0).cpu().numpy()  # 1, 100
    else:
        raise NotImplementedError
    return secret_pred


@st.cache_resource
def load_model(model_name):
    if model_name == 'RoSteALS':
        # prep = transforms.Resize((256,256))  # preprocess step for display purpose
        # prep = Resize((256,256)) 
        prep = transforms.Resize((256,256))
        tform = transforms.Compose([
            prep,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        model = load_RoSteALS()
    else:
        raise NotImplementedError
    return model, prep, tform

@st.cache_resource
def load_ecc(ecc_name):
    if ecc_name == 'BCH':
        ecc = BCH(payload_len= SECRET_LEN, verbose=True)
    elif ecc_name == 'RSC':
        ecc = RSC(data_bytes=16, ecc_bytes=4, verbose=True)
    return ecc


class Resize(object):
    def __init__(self, size=None) -> None:
        self.size = size
    def __call__(self, x, size=None):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        new_size = size if size is not None else self.size
        if min(x.size) > min(new_size):  # downsample
            x = x.resize(new_size, Image.LANCZOS)
        else:  # upsample
            x = x.resize(new_size, Image.BILINEAR)
        x = np.array(x)
        return x 


def app():
    st.title('Watermarking Demo')
    # setup model
    model_name = st.selectbox("Choose the model", model_names)
    model, prep, tform = load_model(model_name)
    display_width = 300

    # ecc
    ecc = load_ecc('BCH')
    assert ecc.get_total_len() == SECRET_LEN

    # setup st
    st.subheader("Input")
    image_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
    if image_file is not None:
        print('Image: ', image_file.name)
        ext = image_file.name.split('.')[-1]
        im = Image.open(image_file).convert('RGB')
        st.image(im, width=display_width)
    secret_text = st.text_input(f'Input the secret (max {ecc.data_len} chars)', 'secrets')
    assert len(secret_text) <= ecc.data_len, f'Error! Secret length must be smaller than {ecc.data_len//7} ASCII characters'

    # embed
    st.subheader("Embed results")
    status = st.empty()
    if image_file is not None and secret_text is not None:
        secret = ecc.encode_text([secret_text])  # (1, len)
        secret = torch.from_numpy(secret).float()
        stego = embed_secret(model_name, model, im, tform, secret)
        st.image(stego, width=display_width)

        # download button
        mime='image/jpeg' if ext=='jpg' else f'image/{ext}'
        stego_bytes = to_bytes(stego, mime)
        st.download_button(label='Download image', data=stego_bytes, file_name=f'stego.{ext}', mime=mime)

        # verify secret
        secret_pred = decode_secret(model_name, model, Image.fromarray(stego), tform)
        bit_acc = (secret_pred == secret.cpu().numpy()).mean()
        secret_pred = ecc.decode_text(secret_pred)[0]
        status.markdown('**Secret recovery check:** ' + secret_pred, unsafe_allow_html=True)
        status.markdown('**Bit accuracy:** ' + str(bit_acc), unsafe_allow_html=True)

if __name__ == '__main__':
    app()
    

    