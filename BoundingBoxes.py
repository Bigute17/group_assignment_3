import json
from PIL import Image, ImageDraw
import os
import shutil

# Set Image Directory (change this to accomodate yourself)
main_dir = '/Users/blakecurtsinger/Desktop/BZAN554/Group_Assignments/group_assignment_3/SVHN'
train_dir = main_dir + '/train'
test_dir = main_dir + '/test'

os.makedirs(main_dir + "/train_cropped")
os.makedirs(main_dir + "/test_cropped")


# Open the JSON file for reading
with open(train_dir + '/digitStruct.json', 'r') as file:
    # Load the JSON data from the file
    json_data = json.load(file)

for item in json_data:
    filename = item["filename"]  # Extract filename
    boxes = item["boxes"]  # Extract boxes

    # Open the image using PIL
    img = Image.open(train_dir + "/" + filename)

    # Convert the image to RGBA mode
    img = img.convert("RGBA")

    # Create a blank canvas to merge the boxes
    merged_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Paste each box onto the merged image
    for box in boxes:
        left = int(box["left"])
        top = int(box["top"])
        width = int(box["width"])
        height = int(box["height"])

        # Extract the box region from the image
        box_img = img.crop((left, top, left + width, top + height))

        # Paste the box onto the merged image at the corresponding position
        merged_img.paste(box_img, (left, top))

    # Save the merged image to the train_cropped folder
    output_filename = os.path.join(main_dir, "train_cropped", filename)
    merged_img.save(output_filename)



# Open the JSON file for reading
with open(test_dir + '/digitStruct.json', 'r') as file:
    # Load the JSON data from the file
    json_data = json.load(file)

for item in json_data:
    filename = item["filename"]  # Extract filename
    boxes = item["boxes"]  # Extract boxes

    # Open the image using PIL
    img = Image.open(train_dir + "/" + filename)

    # Convert the image to RGBA mode
    img = img.convert("RGBA")

    # Create a blank canvas to merge the boxes
    merged_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Paste each box onto the merged image
    for box in boxes:
        left = int(box["left"])
        top = int(box["top"])
        width = int(box["width"])
        height = int(box["height"])

        # Extract the box region from the image
        box_img = img.crop((left, top, left + width, top + height))

        # Paste the box onto the merged image at the corresponding position
        merged_img.paste(box_img, (left, top))

    # Save the merged image to the train_cropped folder
    output_filename = os.path.join(main_dir, "test_cropped", filename)
    merged_img.save(output_filename)
