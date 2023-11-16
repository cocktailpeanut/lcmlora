from diffusers import DiffusionPipeline, LCMScheduler, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline, ControlNetModel
import cv2
import numpy as np
from PIL import Image

import torch
import gradio as gr
import time

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
#model_id = "runwayml/stable-diffusion-v1-5"
#lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

if torch.backends.mps.is_available():
  DEVICE = "mps"
elif torch.cuda.is_available():
  DEVICE = "cuda"
else:
  DEVICE = "cpu"


#print(f"device={DEVICE}")
#
## Txt2Img Pipe
#pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")
#pipe.load_lora_weights(lcm_lora_id)
#pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#pipe.to(device=DEVICE, dtype=torch.float16)
#
#
#
#
## Img2Img Pipe2
#pipe2 = AutoPipelineForImage2Image.from_pretrained(model_id, variant="fp16")
#pipe2.load_lora_weights(lcm_lora_id)
#pipe2.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
#pipe2.to(device=DEVICE, dtype=torch.float16)
##pipe2.enable_model_cpu_offload()

# Controlnet Pipe3
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0-small", torch_dtype=torch.float16, variant="fp16")
pipe3 = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    variant="fp16"
)
pipe3.to(device=DEVICE, dtype=torch.float16)
pipe3.load_lora_weights(lcm_lora_id)
pipe3.scheduler = LCMScheduler.from_config(pipe3.scheduler.config)
pipe3.fuse_lora()


lastPrompt = ""
lastTs = time.time()
running = False
image = None

def controlnet(prompt, image):
  
  # canny_image
  low_threshold = 100
  high_threshold = 200

  #image = np.array(image.resize((1024,1024)))


  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)

  generator = torch.manual_seed(0)
  image = pipe3(
      prompt,
      image=canny_image,
      num_inference_steps=5,
      guidance_scale=1.5,
      controlnet_conditioning_scale=0.5,
      cross_attention_kwargs={"scale": 1},
      generator=generator,
  ).images[0]

  return image



def img2img(prompt, image):
  image = pipe2(
      prompt=prompt,
      image=image,
      num_inference_steps=4,
      guidance_scale=1,
  ).images[0]
  return image


def txt2img(prompt):
  global image
  global running

  if running:
    return image
  else:
    running = True
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=1,
    ).images[0]
    running = False
    return image

demo = gr.Interface(
    fn=controlnet,
    inputs=[
        gr.Textbox(value="a cat in the hat"),
        gr.Image()
    ],
    outputs=gr.Image()
)
demo.launch()
