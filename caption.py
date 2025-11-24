from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def describe(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_length=80,
        min_length=10,
        num_beams=8,
        length_penalty=1.5,
        repetition_penalty=1.2,
    )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

