import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from thai_nner import NNER

nner = NNER("model.pth")
tags = nner.get_tag("วันนี้วันที่ 5 เมษายน 2565 เป็นวันที่อากาศดีมาก")
print(tags)