import torch
import numpy as np
import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageDraw

import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

def crop_and_resize_text_area(
    image,
    text_bbox,
    original_coordinates=None,
    word_crop_margin=1.0,
    size=256,
    allowed_magnification=5,
):
    cropped_img = image.copy()

    text_area_witdh = text_bbox[2] - text_bbox[0]
    text_area_height = text_bbox[3] - text_bbox[1]

    text_area_size = max(text_area_witdh, text_area_height) * (1 + word_crop_margin)
    # テキスト範囲が画像をはみ出てしまった場合は修正
    if text_area_size > max(image.size):
        text_area_size = max(image.size)

    text_center = np.array(
        [(text_bbox[0] + text_bbox[2]) / 2, (text_bbox[1] + text_bbox[3]) / 2]
    )

    # 切り取る範囲が画像の範囲を逸脱していた場合に範囲を修正
    def modify_crop_area(crop_area):
        if image.width > (size / allowed_magnification):
            if crop_area[0, 0] < 0:
                crop_area[:, 0] -= crop_area[0, 0]
            elif crop_area[1, 0] > image.width:
                crop_area[:, 0] -= crop_area[1, 0] - image.width
        if image.height > (size / allowed_magnification):
            if crop_area[0, 1] < 0:
                crop_area[:, 1] -= crop_area[0, 1]
            elif crop_area[1, 1] > image.height:
                crop_area[:, 1] -= crop_area[1, 1] - image.height

    # テキスト範囲がsize/拡大許可倍率よりも大きい場合はテキスト範囲でcrop、小さい場合はsize/拡大許可倍率でcrop
    text_area_size = max([text_area_size, size / allowed_magnification])
    size_array = np.array([[-text_area_size / 2] * 2, [text_area_size / 2] * 2])
    crop_area = text_center + size_array
    modify_crop_area(crop_area)

    crop_area = crop_area.astype(int)

    cropped_img = cropped_img.crop(tuple(crop_area.ravel()))
    cropped_img = cropped_img.resize((size, size))

    # 画像変換後の座標を計算
    if original_coordinates is None:
        transformed_coordinates = None
    else:
        transformed_coordinates = (
            (original_coordinates - crop_area[0]) * size / text_area_size
        ).astype(int)

    return cropped_img, transformed_coordinates, crop_area


#### import diffusion models
text_encoder = CLIPTextModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="text_encoder"
).cuda().half()
tokenizer = CLIPTokenizer.from_pretrained(
    'sd-legacy/stable-diffusion-v1-5', subfolder="tokenizer"
)

#### additional tokens are introduced, including coordinate tokens and character tokens
print('***************')
print(len(tokenizer))
for i in range(520):
    tokenizer.add_tokens(['l' + str(i) ]) # left
    tokenizer.add_tokens(['t' + str(i) ]) # top
    tokenizer.add_tokens(['r' + str(i) ]) # width
    tokenizer.add_tokens(['b' + str(i) ]) # height    
for c in alphabet:
    tokenizer.add_tokens([f'[{c}]']) 
print(len(tokenizer))
print('***************')

vae = AutoencoderKL.from_pretrained('sd-legacy/stable-diffusion-v1-5', subfolder="vae").half().cuda()
unet = UNet2DConditionModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft-inpainting', subfolder="unet"
).half().cuda()
text_encoder.resize_token_embeddings(len(tokenizer))


def to_tensor(image):
    if isinstance(image, Image.Image):  
        image = np.array(image)
    elif not isinstance(image, np.ndarray):  
        raise TypeError("Error")

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image)

    return tensor

def tuple_to_ndarray(tpl):
    if len(tpl) % 2 != 0:
        raise ValueError("Tuple must contain an even number of elements")
    return np.array(tpl, dtype=int).reshape(-1, 2)

def ndarray_to_tuple(arr):
    if arr.shape[1] != 2:
        raise ValueError("Array must have shape (N, 2)")
    return tuple(arr.flatten())

i = "./img_1.jpg"

orig_i = "./img_1.jpg"
position = (38, 43, 920, 215)
text = "Requirement"

cropped_img, transformed_coordinates, crop_area = crop_and_resize_text_area(
    Image.open(orig_i),
    text_bbox=position,
    original_coordinates=tuple_to_ndarray(position),
    size=512,
    allowed_magnification=5
)

position = ndarray_to_tuple(transformed_coordinates)

prompt = ""

step = 20
guidance = 2.5
batch = 1
temperature = 1.4


with torch.no_grad():
    time1 = time.time()
    user_prompt = prompt
    
    user_prompt += ' <|endoftext|><|startoftext|>'
    layout_image = None

    image_mask = Image.new('L', (512,512), 0)
    draw = ImageDraw.Draw(image_mask)
    
    if len(position) == 2:
        x, y = position
        x = x // 4
        y = y // 4
        text_str = ' '.join([f'[{c}]' for c in list(text)])
        user_prompt += f' l{x} t{y} {text_str} <|endoftext|>'

    elif len(position) == 4:
        x0, y0, x1, y1 = position
        x0 = x0 // 4
        y0 = y0 // 4
        x1 = x1 // 4
        y1 = y1 // 4
        text_str = ' '.join([f'[{c}]' for c in list(text)])
        user_prompt += f' l{x0} t{y0} r{x1} b{y1} {text_str} <|endoftext|>'

        draw.rectangle((x0*4, y0*4, x1*4, y1*4), fill=1)
        print('prompt ', user_prompt)

    elif len(position) == 8: # four points
        x0, y0, x1, y1, x2, y2, x3, y3 = position
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=1)
        x0 = x0 // 4
        y0 = y0 // 4
        x1 = x1 // 4
        y1 = y1 // 4
        x2 = x2 // 4
        y2 = y2 // 4
        x3 = x3 // 4
        y3 = y3 // 4
        xmin = min(x0, x1, x2, x3)
        ymin = min(y0, y1, y2, y3)
        xmax = max(x0, x1, x2, x3)
        ymax = max(y0, y1, y2, y3)
        text_str = ' '.join([f'[{c}]' for c in list(text)])
        user_prompt += f' l{xmin} t{ymin} r{xmax} b{ymax} {text_str} <|endoftext|>'

        print('prompt ', user_prompt)


    prompt = tokenizer.encode(user_prompt)
    composed_prompt = tokenizer.decode(prompt)

    prompt = prompt[:77]
    while len(prompt) < 77: 
        prompt.append(tokenizer.pad_token_id) 

    prompts_cond = prompt
    prompts_nocond = [tokenizer.pad_token_id]*77

    prompts_cond = [prompts_cond] * batch
    prompts_nocond = [prompts_nocond] * batch

    prompts_cond = torch.Tensor(prompts_cond).long().cuda()
    prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

    scheduler = DDPMScheduler.from_pretrained('sd-legacy/stable-diffusion-v1-5', subfolder="scheduler") 
    scheduler.set_timesteps(step) 
    noise = torch.randn((batch, 4, 64, 64)).to("cuda").half()
    input = noise

    encoder_hidden_states_cond = text_encoder(prompts_cond)[0].half()
    encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0].half()

    image_mask = torch.Tensor(np.array(image_mask)).float().half().cuda()
    image_mask = image_mask.unsqueeze(0).unsqueeze(0).repeat(batch, 1, 1, 1)

    image = cropped_img
    image_tensor = to_tensor(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)   
    print(f'image_tensor.shape {image_tensor.shape}')
    masked_image = image_tensor * (1-image_mask)
    masked_feature = vae.encode(masked_image.half()).latent_dist.sample() 
    masked_feature = masked_feature * vae.config.scaling_factor
    masked_feature = masked_feature.half()
    print(f'masked_feature.shape {masked_feature.shape}')

    feature_mask = torch.nn.functional.interpolate(image_mask, size=(64,64), mode='nearest').cuda()

    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():  # classifier free guidance

            noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:batch],feature_mask=feature_mask, masked_feature=masked_feature).sample # b, 4, 64, 64
            noisy_residual = noise_pred_uncond + guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
            input = scheduler.step(noisy_residual, t, input).prev_sample
            del noise_pred_cond
            del noise_pred_uncond

            torch.cuda.empty_cache()

    # decode
    input = 1 / vae.config.scaling_factor * input 
    images = vae.decode(input, return_dict=False)[0] 
    width, height = 512, 512
    results = []
    new_image = Image.new('RGB', (2*width, 2*height))
    for index, image in enumerate(images.cpu().float()):
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
        results.append(image)
        row = index // 2
        col = index % 2
        new_image.paste(image, (col*width, row*height))

    # os.system('nvidia-smi')
    torch.cuda.empty_cache()
    # os.system('nvidia-smi')

    results[0].save("output.png")
    print(composed_prompt)
