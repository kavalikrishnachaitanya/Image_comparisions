# import numpy as np
# import torchvision.models as models
# import torch
# from PIL import Image

# # Load the two images
# image1 = torch.from_numpy(np.array(Image.open("image1.jpeg"))).float()
# image2 = torch.from_numpy(np.array(Image.open("image2.jpeg"))).float()


# # Get the VGG16 model
# vgg16 = models.vgg16(pretrained=True)

# # Extract the features of the two images
# features1 = vgg16.features[0](image1)
# features2 = vgg16.features[0](image2)

# # Calculate the similarity between the features of the two images
# similarity = torch.cosine_similarity(features1, features2)

# # Determine the byproduct
# byproduct = image1 if similarity[0] > similarity[1] else image2

# # Save the byproduct
# Image.fromarray(byproduct.cpu().numpy()).save("byproduct.jpg")


import numpy as np
import torchvision.models as models
import torch
from PIL import Image

# Load the two images
image1 = torch.from_numpy(np.array(Image.open("image1.jpeg"))).float().resize_(224, 224).view(224, 224)
image2 = torch.from_numpy(np.array(Image.open("image2.jpeg"))).float().resize_(224, 224).view(224, 224)


# Get the VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Extract the features of the two images
features1 = vgg16.features[0](image1)
features2 = vgg16.features[0](image2)

# Calculate the similarity between the features of the two images
similarity = torch.cosine_similarity(features1, features2)

# Determine the byproduct
byproduct = image1 if similarity[0] > similarity[1] else image2

# Save the byproduct
Image.fromarray(byproduct.cpu().numpy()).save("byproduct.jpg")
