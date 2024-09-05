from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionXLPipeline

model_name='/dataset/CommonCanvas-XL-C'

scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    use_habana=True,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

outputs = pipeline(
    ["A panda eating a taco"],
    num_images_per_prompt=8,
    batch_size=4,
)
image_save_dir="."
for i, image in enumerate(outputs.images):
    image.save( f"image_{i+1}.png")

print('End!')
