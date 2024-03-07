from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import os
import math
import numpy as np
import argparse
import json

def place_text_on_image(image_path, output_path, text,
                        font_size=40, x=100, y=100,
                        r=255, g=255, b=255, angle=0):
    '''Writes passed in text on the given image.
    Args:
      image_path: string for input path; (single image)
      output_path: string for output path; (directory)
      text: string of text that will be placed on images

    Returns:
      None
    '''
    # initialize output folder if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with Image.open(image_path).convert("RGBA") as img:
        # resize the image first to make it a square
        # use bicubic resampling to be consistent with Bunny's evaclip processor bicubic resizing method
        img = img.resize((384, 384), resample = Image.Resampling.BICUBIC)
        
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        width, height = img.size
        
        font_path = "font_file.ttf"

        font = ImageFont.truetype(font_path, font_size)
        text_length = draw.textlength(text, font=font)

        # create a new blank canvas
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.text((x,y), text, fill=(r,g,b), font=font, font_size=font_size)

        # rotate the overlay with the text
        overlay = overlay.rotate(angle)

        # paste the rotated overlay onto the original image
        img = Image.alpha_composite(img, overlay)

        # save the image with text
        base, extension_type = os.path.splitext(image_path)
        # change file extension type to PNG for RGBA mode to work
        if extension_type == ".png":
            new_image_path = image_path
        else:
            new_image_path = base + '.png'
        
        new_image_path = f"with_text_{os.path.basename(new_image_path)}"
        # new image path to output directory
        new_image_path = os.path.join(output_path, new_image_path)
        img.save(new_image_path)
        print(f"Saved {new_image_path}")

    
    #save dataset info to json file
    json_dict = {}
    json_dict["text"] = text
    json_dict["font_size"] = font_size
    json_dict["x"] = x
    json_dict["y"] = y
    json_dict["r"] = r
    json_dict["g"] = g
    json_dict["b"] = b
    json_dict["angle"] = angle
    
    file_name = os.path.join(output_path, "info.json")
    with open(file_name, 'w') as file:
        json.dump(json_dict, file, indent=4)


def run(args, fontsize=False, rotation=False):
    '''Runs place_text_on_image for all images in the input_path in args.'''
    
    # save list of images and texts
    texts = ['cheetah']
    print(f"texts: {texts}")

    # insert your image directory
    input_path = args.input_path
    output_path = args.output_path
    images = os.listdir(input_path)
    images = [os.path.join(input_path, img_path) for img_path in images if not img_path.startswith('.')]

    # loop through each image and place random text on it
    for image_path in images:
        random_text = random.choice(texts)
        if fontsize:
            sizes = np.arange(10, 100, 10)
            for size in sizes:
                place_text_on_image(image_path, output_path, random_text, font_size=size)
        if rotation:
            rotations = np.arange(0, 360, 45)
            for rotation in rotations:
                place_text_on_image(image_path, output_path, random_text, angle=rotation)
        else:
            place_text_on_image(image_path, output_path, random_text)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Add text to images in selected directory"
    )

    parser.add_argument(
        "-i",
        "--input_path",
        type = str,
        default = "bunny/serve/examples",
        help = "Input directory where all images are stored"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type = str,
        default="bunny/serve/text_outputs",
        help = "output directory where all output images are stored"
    )

    args = parser.parse_args()

    run(args, fontsize=True, rotation=False)
    print("Finished")