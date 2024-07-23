import sys
sys.path.append('CodeFormer')
import os
import base64
from io import BytesIO
import requests
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image


class InferlessPythonModel:
    def initialize(self):
        self.device = torch.device('cuda')
        nfs_volume = os.getenv("NFS_VOLUME")
        
        if os.path.exists(nfs_volume + "codeformer.pth") == False :
            os.system(f"wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -P {nfs_volume}")
            os.system(f"wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth -P {nfs_volume}")
        

        model = RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=2,)
        self.upsampler = RealESRGANer(scale=2,model_path=f"{nfs_volume}/RealESRGAN_x2plus.pth",model=model,tile=400,tile_pad=40,pre_pad=0,half=True)
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=["32", "64", "128", "256"]).to(self.device)
        checkpoint = torch.load(f"{nfs_volume}/codeformer.pth")["params_ema"]
        self.codeformer_net.load_state_dict(checkpoint)
        self.codeformer_net.eval()
    
    def download_img(self,url,filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename

    def infer(self, inputs):
        img_url = inputs["img_url"]
        codeformer_fidelity = inputs["codeformer_fidelity"]
        face_align = inputs.get("face_align",True)
        background_enhance = inputs.get("background_enhance",True)
        face_upsample = inputs.get("face_upsample",True)
        upscale = inputs.get("upscale",2)

        img = self.download_img(img_url,"temp.jpg")

        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"
        face_align = face_align if face_align is not None else True
        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        has_aligned = not face_align
        upscale = 1 if has_aligned else upscale

        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(upscale,face_size=512,crop_ratio=(1, 1),det_model=detection_model,save_ext="png",use_parse=True,device=self.device)
        bg_upsampler = self.upsampler if background_enhance else None
        face_upsampler = self.upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )
        else:
            restored_img = restored_face
        
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(restored_img)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        base64_image = base64.b64encode(buff.getvalue()).decode('utf-8')

        return {"generated_image":base64_image}
    
    def finalize(self):
       self.face_enhancer = None