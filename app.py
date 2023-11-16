from diffusers import DiffusionPipeline, LCMScheduler
import torch
import gradio as gr

#model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "justinpinkney/miniSD"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

if torch.backends.mps.is_available():
  DEVICE = "mps"
elif torch.cuda.is_available():
  DEVICE = "cuda"
else:
  DEVICE = "cpu"


print(f"device={DEVICE}")
pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16")

pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device=DEVICE, dtype=torch.float16)


def generate(prompt):
  images = pipe(
      prompt=prompt,
      height=256,
      width=256,
      num_inference_steps=4,
      guidance_scale=1,
  ).images[0]
  return images

demo = gr.Interface(fn=generate, inputs=gr.Textbox(value="a cat in the hat"), outputs="image")
demo.launch()
