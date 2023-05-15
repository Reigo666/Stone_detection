import json

import requests
from PIL import Image
import numpy as np
import base64

image = Image.open(r'D:\jpf\segment\coco\test15\JPEGImages\aggregate_2022-10-11-10-32-50_1_c.jpg')
image = np.array(image)
image = image.tolist()
proxies = { "http": None, "https": None}
res = requests.post(url='http://192.168.16.40:5000/stone',
                    headers={"Content-Type": "application/json"},
                    json={"image": image,
                          "type": "image",
                          "sharding_id": 0,
                          },
                    proxies=proxies
                    )
