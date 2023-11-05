import json
import os
import sys
import glob
import pickle
import random

import numpy as np

import cv2 as cv
import torch
from torchvision import transforms

from imageencoder import encoder_init, encode_real_images
from DPR import dpr_init, get_lightvec

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

landmarks_model_path = "../mymodels/shape_predictor_68_face_landmarks.dat"
landmarks_detector = LandmarksDetector(landmarks_model_path)

def getface(imagepath):
    face_landmarks = landmarks_detector.get_landmarks(imagepath)
    face_landmarks = list(face_landmarks)
    if len(face_landmarks) == 0:
        return None
    img = image_align(imagepath, None, face_landmarks[0], output_size=1024)
    return img

name_list = ['beauty']

expression_dict = {0: 'none', 1: 'smile', 2: 'laugh'}
face_shape_dict = {0: 'square', 1: 'oval', 2: 'heart', 3: 'round', 4: 'triangle'}
face_type_dict = {0: 'human', 1: 'cartoon'}
gender_dict = {0: 'female', 1: 'male'}
glasses_dict = {0: 'none', 1: 'sun', 2: 'common'}
race_dict = {0: 'yellow', 1: 'white', 2: 'black', 3: 'arabs'}

def idx2name(idx, tag):
    name = None
    if tag == 'expression':
        name = expression_dict[idx]
    elif tag == 'face_shape':
        name = face_shape_dict[idx]
    elif tag == 'face_type':
        name = face_type_dict[idx]
    elif tag == 'gender':
        name = gender_dict[idx]
    elif tag == 'glasses':
        name = glasses_dict[idx]
    elif tag == 'race':
        name = race_dict[idx]
    return name

def name2idx(name):
    lookup_table = {'none': 0, 'smile': 1, 'laugh': 2,
                    'square': 0, 'oval': 1, 'heart': 2, 'round': 3, 'triangle': 4,
                    'human': 0, 'cartoon': 1,
                    'female': 0, 'male': 1,
                    'sun': 1, 'common': 2,
                    'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}

    return lookup_table[name]

if __name__ == "__main__":
    samples = glob.glob(sys.argv[1]+"/*.*")

    device = 'cuda'

    encoder, G, dlatent_avg = encoder_init(device)

    lightmodel = dpr_init(device)

    checkpoint = '../mymodels/face-attributes_scripted.pt'
    faceattr_model = torch.jit.load(checkpoint, map_location=device)
    faceattr_model = faceattr_model.to(device)
    faceattr_model.eval()
    faceattr_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dlatents_arr = np.zeros((len(samples),1,18,512), dtype=np.float32)
    faceattr_arr = np.zeros((len(samples),8,1), dtype=np.float32)
    lights = np.zeros((len(samples),1,9,1,1), dtype=np.float32)

    for i, full_path in enumerate(samples):
        print(full_path)

        faceimg = getface(full_path)

        if faceimg is None:
            continue

        faceimg = np.array(faceimg)

        # encode image 
        dlatents = encode_real_images(
            device, G, encoder, dlatent_avg, faceimg, truncation_psi=0.5, num_steps=1
            )
        dlatents_arr[i,0,:,:] = dlatents

        out_light = get_lightvec(device, lightmodel, faceimg)
        lights[i,:,:,:,:] = out_light.detach().cpu().numpy()

        # xxx: accuracy of age and eyeglasses is very poor
        #faceimg = cv.resize(faceimg, (224,224))
        #faceimg = faceimg[..., ::-1]  # RGB
        #faceimg = transforms.ToPILImage()(faceimg)
        faceimg = faceimg.resize((224,224))
        faceimg = faceattr_trans(faceimg)
        inputs = torch.unsqueeze(faceimg, 0).float().to(device)

        with torch.no_grad():
            reg_out, expression_out, gender_out, glasses_out, race_out = faceattr_model(inputs)

            reg_out = reg_out.cpu().numpy()
            age_out = reg_out[:, 0]
            pitch_out = reg_out[:, 1]
            roll_out = reg_out[:, 2]
            yaw_out = reg_out[:, 3]
            beauty_out = reg_out[:, 4]

            _, expression_out = expression_out.topk(1, 1, True, True)
            _, gender_out = gender_out.topk(1, 1, True, True)
            _, glasses_out = glasses_out.topk(1, 1, True, True)
            _, race_out = race_out.topk(1, 1, True, True)

            expression_out = expression_out.cpu().numpy()
            gender_out = gender_out.cpu().numpy()
            glasses_out = glasses_out.cpu().numpy()
            race_out = race_out.cpu().numpy()

            age = int(age_out[0] * 100)
            pitch = float('{0:.2f}'.format(pitch_out[0] * 360 - 180))
            roll = float('{0:.2f}'.format(roll_out[0] * 360 - 180))
            yaw = float('{0:.2f}'.format(yaw_out[0] * 360 - 180))
            beauty = float('{0:.2f}'.format(beauty_out[0] * 100))
            expression = idx2name(int(expression_out[0][0]), 'expression')
            gender = idx2name(int(gender_out[0][0]), 'gender')
            glasses = idx2name(int(glasses_out[0][0]), 'glasses')
            race = idx2name(int(race_out[0][0]), 'race')

            # 23 18.3 1.65 33.47 none male none white
            print(age, pitch, yaw, beauty, expression, gender, glasses, race)

            # ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
            faceattr_arr[i,0,0] = 0. if gender == 'female' else 1.
            faceattr_arr[i,1,0] = 0. if glasses == 'none' else 1.
            faceattr_arr[i,2,0] = max(-20,min(yaw,20))
            faceattr_arr[i,3,0] = max(-20,min(pitch,20))
            faceattr_arr[i,4,0] = 0.5
            faceattr_arr[i,5,0] = 0.5
            faceattr_arr[i,6,0] = min(age_out[0],65)
            faceattr_arr[i,7,0] = name2idx(expression)

    np.save("./data/dlatents.npy", dlatents_arr)
    np.save("./data/attributes.npy", faceattr_arr)
    np.save("./data/lights.npy", lights)
