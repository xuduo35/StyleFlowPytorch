# StyleFlowPytorch
Remove tensorflow dependence of official StyleFlow, and support specific image editing

# Borrow code from
https://github.com/RameenAbdal/StyleFlow

https://github.com/adriansahlman/stylegan2_pytorch

https://github.com/eladrich/pixel2style2pixel

https://github.com/zhhoper/DPR

https://github.com/foamliu/Face-Attributes-Mobile

All borrowed code are subject to original license.

# Usage
Download Gs_ffhq.pth from https://drive.google.com/file/d/1YVGoe2b5nj1ogUtS5kYS8ptehAa4K8Km/view?usp=sharing, it's convertd from stylegan2-ffhq-config-f.pkl using adriansahlman's code. Put it under mymodels.

Put 'psp_encoder.pth'(pixel2style2pixel) under mymodels. Can be downloaded from https://drive.google.com/file/d/1vDfvBDFXXY4CIaJH4P0AhponaN4FzM9P/view?usp=sharing.

Put 'shape_predictor_68_face_landmarks.dat'(dlib) under mymodels.

cd webui

Use random generate images:

CUDA_VISIBLE_DEVICES=0 streamlit run app.py

Or use your own images:

python3 gendata.py ./images

CUDA_VISIBLE_DEVICES=0 streamlit run app.py ./data

Just like https://github.com/RameenAbdal/StyleFlow webui

You can edit(or create) ~/.streamlit/config.toml  file to config port. Including content like:
[server]
port=8888
