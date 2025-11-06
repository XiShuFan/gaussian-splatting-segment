# Imports
from PIL import Image
import torch
from torchvision import transforms

from models.birefnet import BiRefNet

model_name = ['BiRefNet', 'BiRefNet_HR', 'BiRefNet_HR-matting'][0]

# # Option 1: loading BiRefNet with weights:
# from transformers import AutoModelForImageSegmentation
# birefnet = AutoModelForImageSegmentation.from_pretrained('./ckpt/{}'.format(model_name), trust_remote_code=True)

# Option-2: loading weights with BiReNet codes:
birefnet = BiRefNet.from_pretrained('/media/why/新加卷/xsf/BiRefNet/ckpt/BiRefNet')

# # Option-3: Loading model and weights from local disk:
# from utils import check_state_dict

# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load('../BiRefNet-general-epoch_244.pth', map_location='cpu')
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)


# Load Model
device = 'cuda'
torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')
birefnet.half()

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024) if '_HR' not in model_name else (2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import os
from glob import glob
from image_proc import refine_foreground

def main(src_dir, dst_dir):
    image_paths = glob(os.path.join(src_dir, '*'))
    os.makedirs(dst_dir, exist_ok=True)
    for image_path in image_paths:
        print('Processing {} ...'.format(image_path))
        image = Image.open(image_path)
        input_images = transform_image(image).unsqueeze(0).to('cuda')
        input_images = input_images.half()

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Save Results
        file_ext = os.path.splitext(image_path)[-1]
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)
        pred_pil.save(image_path.replace(src_dir, dst_dir).replace(file_ext, '.png'))
        # image_masked = refine_foreground(image, pred_pil)
        # image_masked.putalpha(pred_pil)
        # image_masked.save(image_path.replace(src_dir, dst_dir).replace(file_ext, '-subject.png'))

    print("done")
    

if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--dst_dir', type=str)
    args: Namespace = parser.parse_args()
    main(args.src_dir, args.dst_dir)