import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

import time


#url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
filename = "./IMG_6041.jpg"
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)


with torch.no_grad():
    start = time.time()
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    end = time.time()
    print("time of inference:", end - start)

#z = torch.exp(prediction/5)
#cv2.imwrite("tmp.png", z.cpu().numpy())
#prediction = torch.exp(prediction / 5)

z = torch.mean(prediction, dim=0)
z = z.unsqueeze(0)
#kernel = torch.ones((1, 1, 5))
#z = torch.nn.functional.conv1d(z.unsqueeze(0), kernel, padding="same")
sz = z.shape[1]
nsz = 50
z = torch.nn.functional.adaptive_avg_pool1d(z, nsz).squeeze()
i = torch.argmin(z)
il = int(i / nsz * sz)
ir = int((i+1) / nsz * sz)
prediction[:, il:ir] += 10000
"""
def compute_centroid(signal):
    x = torch.arange(signal.shape[0])
    total_w = torch.sum(signal)
    if total_w == 0:
        return 0.5
    centroid = torch.sum(x * signal) / total_w
    return centroid 
i = compute_centroid(z)
print(i)
i = int(i)
prediction[:, (i-10):(i+10)] += 10000
"""
out_img = prediction.cpu().numpy()
cv2.imwrite(f"out/{filename}.png", out_img)