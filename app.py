%%writefile app.py
import os
import sys
import cv2
import torch
import streamlit as st
import numpy as np
from torchvision.transforms import functional
from PIL import Image
sys.modules["torchvision.transforms.functional_tensor"] = functional
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer
import tempfile
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Initialize face analysis app
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Load the face swapper model
swapper = get_model('/content/drive/MyDrive/CodeFileShukdev/Myfiles/inswapper_128.onnx', download=False, download_zip=False)

# Download Required Models for enhancement
if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")
if not os.path.exists('GFPGANv1.2.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth -P .")
if not os.path.exists('GFPGANv1.3.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P .")
if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('RestoreFormer.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth -P .")

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

def swap_faces(img1, img2):
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]

    img1_swapped = swapper.get(img1, face1, face2, paste_back=True)
    img2_swapped = swapper.get(img2, face2, face1, paste_back=True)

    return img1_swapped, img2_swapped

def upscaler(img, version, scale):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_enhancer = GFPGANer(
            model_path=f'{version}.pth',
            upscale=2,
            arch='RestoreFormer' if version == 'RestoreFormer' else 'clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        h, w = output.shape[0:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return output
    except Exception as error:
        st.error(f"Error occurred: {error}")
        return None

def main():
    st.title("Face Swap and Enhance App")

    # Upload source image
    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    # Upload target image
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

    if source_image is not None and target_image is not None:
        # Read images
        img1 = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), 1)
        img2 = cv2.imdecode(np.frombuffer(target_image.read(), np.uint8), 1)

        # Swap faces
        img1_swapped, img2_swapped = swap_faces(img1, img2)

        # Display original and swapped images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Source Image")
            st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), channels="RGB")

        with col2:
            st.subheader("Target Image")
            st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), channels="RGB")

        st.subheader("Swapped Images")
        col3, col4 = st.columns(2)

        with col3:
            st.image(cv2.cvtColor(img1_swapped, cv2.COLOR_BGR2RGB), channels="RGB", caption="Source with Target Face")
            img1_swapped_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(img1_swapped_temp.name, img1_swapped)

        with col4:
            st.image(cv2.cvtColor(img2_swapped, cv2.COLOR_BGR2RGB), channels="RGB", caption="Target with Source Face")
            img2_swapped_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(img2_swapped_temp.name, img2_swapped)

        # Image enhancement section
        st.subheader("Enhance Swapped Images")
        version = st.radio('Select Version', ['GFPGANv1.2', 'GFPGANv1.3', 'GFPGANv1.4', 'RestoreFormer'])
        scale = st.number_input("Rescaling Factor", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

        if st.button("Enhance the Images"):
            img1_enhanced = upscaler(img1_swapped, version, scale)
            img2_enhanced = upscaler(img2_swapped, version, scale)

            if img1_enhanced is not None:
                st.image(img1_enhanced, caption="Enhanced Source with Target Face", use_column_width=True)
                enhanced_temp_1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(enhanced_temp_1.name, img1_enhanced)
                with open(enhanced_temp_1.name, "rb") as f:
                    st.download_button("Download Enhanced Source with Target Face", f, file_name="enhanced_source_with_target_face.jpg", mime="image/jpeg")

            if img2_enhanced is not None:
                st.image(img2_enhanced, caption="Enhanced Target with Source Face", use_column_width=True)
                enhanced_temp_2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(enhanced_temp_2.name, img2_enhanced)
                with open(enhanced_temp_2.name, "rb") as f:
                    st.download_button("Download Enhanced Target with Source Face", f, file_name="enhanced_target_with_source_face.jpg", mime="image/jpeg")

if __name__ == "__main__":
    main()
