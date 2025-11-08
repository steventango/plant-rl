import glob

from PIL import Image

# Find all alliance zone camera images and sort them
image_files = sorted(glob.glob("alliance-zone*-camera*.jpg"))

# Open images
images = [Image.open(f) for f in image_files]

# Resize each image to 400x300 (maintaining aspect ratio by cropping or fitting)
thumb_size = (400, 300)
thumbs = []
for img in images:
    # Resize to fit within 400x300, maintaining aspect ratio
    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
    # Create a new image with white background
    thumb = Image.new("RGB", thumb_size, (255, 255, 255))
    # Paste the resized image centered
    x = (thumb_size[0] - img.size[0]) // 2
    y = (thumb_size[1] - img.size[1]) // 2
    thumb.paste(img, (x, y))
    thumbs.append(thumb)

# Create collage: 4 columns, 3 rows
cols = 4
rows = 3
collage_width = cols * thumb_size[0]
collage_height = rows * thumb_size[1]
collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))

# Paste images
for i, thumb in enumerate(thumbs):
    row = i // cols
    col = i % cols
    x = col * thumb_size[0]
    y = row * thumb_size[1]
    collage.paste(thumb, (x, y))

# Save the collage
collage.save("alliance_collage.jpg")
print("Collage saved as alliance_collage.jpg")
