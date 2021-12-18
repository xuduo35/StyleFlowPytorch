# Author shariqfarooq123
import os
import sys
sys.path.insert(0, "../")

import cv2
import random
import numpy as np
import pickle
import copy

import torch
import torch.nn
from module.flow import cnf
import stylegan2
from stylegan2 import utils

import streamlit as st

st.set_page_config(
    layout="wide",  # Can be "centered" or "wide"
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="StyleFlow web demo",  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

""" # Welcome to SyleFlow WebUI demo (Beta)
Go wild!
"""


DATA_ROOT = "../data"
HASH_FUNCS = {torch.nn.Module: id,
              torch.nn.Parameter: lambda x: x.detach().cpu().numpy(),
              torch.Tensor: lambda x: x.cpu().numpy()}

# Select images
all_idx = np.array(random.sample(range(1,999), 32))

EPS = 1e-3  # arbitrary positive value

device = 'cuda'

class State:  # Simple dirty hack for maintaining state
    prev_attr = None
    prev_idx = None
    first = True
    # ... and other state variables

if not hasattr(st, 'data'):  # Run only once. Save data globally

    st.state = State()
    with st.spinner("Setting up... This might take a few minutes"):
        if len(sys.argv) > 1:
            raw_w = np.load('{}/dlatents.npy'.format(sys.argv[1]))
            raw_attr = np.load('{}/attributes.npy'.format(sys.argv[1]))
            raw_lights = np.load('{}/lights.npy'.format(sys.argv[1]))
            all_idx = np.array(list(range(0,raw_w.shape[0])), dtype='int')
        else:
            raw_w = pickle.load(open(os.path.join(DATA_ROOT, "sg2latents.pickle"), "rb"))
            raw_w = np.array(raw_w['Latent'])
            # raw_TSNE = np.load(os.path.join(DATA_ROOT, 'TSNE.npy'))  # We are picking images here by index instead
            raw_attr = np.load(os.path.join(DATA_ROOT, 'attributes.npy'))
            raw_lights = np.load(os.path.join(DATA_ROOT, 'light.npy'))

        all_w = raw_w[all_idx]
        all_attr = raw_attr[all_idx]
        all_lights = raw_lights[all_idx]
        print(all_w.shape, all_attr.shape, all_lights.shape)
        pre_lighting = torch.from_numpy(np.load('../mymodels/pre_lighting.npy')).float()

        st.data = dict(
            all_idx=all_idx,
            raw_w=raw_w, all_w=all_w, all_attr=all_attr, all_lights=all_lights,
            pre_lighting=pre_lighting
            )


def make_slider(name, min_value=0.0, max_value=1.0, step=0.1, **kwargs):
    return st.sidebar.slider(name, min_value, max_value, step=step, **kwargs)

@st.cache(allow_output_mutation=True, hash_funcs={dict: id}, show_spinner=False)
def get_idx2init(raw_w, all_idx):
    #idx2init = {i: np.array(raw_w['Latent'])[i] for i in all_idx}
    idx2init = {i: np.array(raw_w)[i] for i in all_idx}
    return idx2init

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_w_to_z(flow_model, w, attributes, lighting):
    w_cuda = torch.Tensor(w)
    att_cuda = torch.from_numpy(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    z = flow_model(w_cuda, features, zero_padding)[0].clone().detach()

    return z

@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def flow_z_to_w(flow_model, z, attributes, lighting):
    att_cuda = torch.Tensor(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    w = flow_model(z, features, zero_padding, True)[0].clone().detach().numpy()

    return w

@st.cache(show_spinner=False, hash_funcs=HASH_FUNCS)
@torch.no_grad()
def generate_image(model, w):
    w = torch.Tensor(w).to(device)
    image_tensor = model.G_synthesis(latents=w)
    image = utils.tensor_to_PIL(image_tensor)[0]
    #image.save('./result.jpg')
    image = np.asarray(image)
    #return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def preserve_w_id(w_new, w_orig, attr_index):
    # Ssssh! secret sauce to strip vectors
    w_orig = torch.Tensor(w_orig)
    if attr_index == 0:
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 1:
        w_new[0][:2] = w_orig[0][:2]
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 2:

        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 3:
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 4:
        w_new[0][6:] = w_orig[0][6:]

    elif attr_index == 5:
        w_new[0][:5] = w_orig[0][:5]
        w_new[0][10:] = w_orig[0][10:]

    elif attr_index == 6:
        w_new[0][0:4] = w_orig[0][0:4]
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 7:
        w_new[0][:4] = w_orig[0][:4]
        w_new[0][6:] = w_orig[0][6:]
    return w_new


def is_new_idx_set(idx):
    if st.state.first:
        st.state.first = False
        st.state.prev_idx = idx
        return True

    if idx != st.state.prev_idx:
        st.state.prev_idx = idx
        return True
    return False

def reset_state(idx):
    st.state = State()
    st.state.first = False
    st.state.prev_idx = idx

def np_copy(*args):  # shortcut to clone multiple arrays
    return [np.copy(arg) for arg in args]

def get_changed_light(lights, light_names):
    for i, name in enumerate(light_names):
        change = abs(lights[name] - st.state.prev_lights[i])
        if change > EPS:
            return i
    return None



def main():
    attribute_names = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]

    light_names = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    att_min = {'Gender': 0, 'Glasses': 0, 'Yaw': -20, 'Pitch': -20, 'Baldness': 0, 'Beard': 0.0, 'Age': 0,
               'Expression': 0}
    att_max = {'Gender': 1, 'Glasses': 1, 'Yaw': 20, 'Pitch': 20, 'Baldness': 1, 'Beard': 1, 'Age': 65, 'Expression': 1}


    with st.spinner("Setting up... This might take a few minutes... Please wait!"):
        all_w, all_attr, all_lights = np_copy(st.data["all_w"], st.data["all_attr"], st.data["all_lights"])
        pre_lighting = list(st.data["pre_lighting"])
        idx2w_init = get_idx2init(st.data["raw_w"], st.data["all_idx"])

    idx_selected = st.selectbox("Choose an image:", list(range(len(idx2w_init))),
                                format_func= lambda opt : st.data["all_idx"][opt])

    w_selected = all_w[idx_selected]
    attr_selected = all_attr[idx_selected].ravel()
    lights_selected = all_lights[idx_selected]
    z_selected = flow_w_to_z(flow_model, w_selected, attr_selected, lights_selected)

    if is_new_idx_set(idx_selected):
        reset_state(idx_selected)
        st.state.prev_attr = attr_selected.copy()
        st.state.prev_lights = lights_selected.ravel().copy()
        st.state.z_current = copy.deepcopy(z_selected)
        st.state.w_current = torch.Tensor(w_selected)
        st.state.w_prev = torch.Tensor(w_selected)
        st.state.light_current = torch.Tensor(lights_selected).float()

    st.sidebar.markdown("# Attributes")
    attributes = {}
    for i, att in enumerate(attribute_names):
        attributes[att] = make_slider(att, float(att_min[att]), float(att_max[att]),
                                      value=float(attr_selected.ravel()[i]),  # value on first render
                                      key=hash(idx_selected*1e5 + i)  # re-render if index selected is changed!
                                      )

    st.sidebar.markdown("# Lighting")
    lights = {}
    for i, lt in enumerate(light_names):
        lights[lt] = make_slider(lt,
                                 value=float(lights_selected.ravel()[i]), # value on first render
                                 key=hash(idx_selected*1e6 + i)  # re-render if index selected is changed!
                                 )

    img_source = generate_image(model, w_selected)

    att_new = list(attributes.values())

    for i, att in enumerate(attribute_names):  # Not the greatest code, but works!
        attr_change = attributes[att] - st.state.prev_attr[i]

        if abs(attr_change) > EPS:
            print(f"Changed attr {att} : {attr_change}")
            attr_final = attr_degree_list[i] * attr_change + st.state.prev_attr[i]
            att_new[i] = attr_final
            print("\n")

            if hasattr(st.state, 'prev_changed') and st.state.prev_changed != att:
                st.state.z_current  = flow_w_to_z(flow_model, st.state.w_current, st.state.prev_attr_factored, lights_selected)
            st.state.prev_attr[i] = attributes[att]
            st.state.prev_changed = att
            st.state.prev_attr_factored = att_new
            st.state.w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lights_selected)
            break  # Streamlit re-runs on each interaction. Probably works but need to test for any bugs here

    pre_lighting_distance = [pre_lighting[i] - st.state.light_current for i in range(len(light_names))]
    lights_magnitude = np.zeros(len(light_names))
    changed_light_index = get_changed_light(lights, light_names)

    if changed_light_index is not None:
        lights_magnitude[changed_light_index] = lights[light_names[changed_light_index]]

        lighting_final = torch.Tensor(st.state.light_current)
        for i in range(len(light_names)):
            lighting_final += lights_magnitude[i] * pre_lighting_distance[i]

        w_current = flow_z_to_w(flow_model, st.state.z_current, att_new, lighting_final)

        w_current[0][0:7] = st.state.w_current[0][0:7] # some stripping
        w_current[0][12:18] = st.state.w_current[0][12:18]

        st.state.w_current = w_current
        lights_new = lighting_final

        st.state.prev_lights[changed_light_index] = lights[light_names[changed_light_index]]
    else:
        lights_new = lights_selected

    col1, col2 = st.beta_columns(2)  # Columns feature of streamlit is still in beta. This line might require to be changed in future versions
    with col1:
        st.image(img_source, caption="Generated", use_column_width=True)

    with col2:
        st.state.w_current = preserve_w_id(st.state.w_current, st.state.w_prev, i)
        img_target = generate_image(model, st.state.w_current)
        st.image(img_target, caption="Target", use_column_width=True)

    st.state.z_current = flow_w_to_z(flow_model, st.state.w_current, att_new, lights_new)
    st.state.w_prev = torch.Tensor(st.state.w_current).clone().detach()

def init_model():
    model = stylegan2.models.load('../mymodels/Gs_ffhq.pth')
    model = utils.unwrap_module(model).to(device)
    model.eval()

    prior = cnf(512, '512-512-512-512-512', 17, 1)
    prior.load_state_dict(torch.load('../flow_weight/modellarge10k.pt'))
    prior.to(device)
    prior.eval()

    return model, prior.cpu()

if __name__ == '__main__':
    # pre_lighting caculated from 8, 33, 641, 547, 28, 34 of ../data/light.npy
    model, flow_model = init_model()

    main()
