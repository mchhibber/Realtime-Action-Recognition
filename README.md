# Realtime-Action-Recognition
Real-time Action Recognition using 3D CNNs for computationally limited platforms. The model developed is based upon MobileNetV2 and is able to perform human action recognition in real-time without GPU requirement.
For model deployment to android app follow <a href="https://github.com/mchhibber/HAR">this</a> link.

## Requirements
<ul><li>Pytorch</li>
<li>Python 3.7+</li>
<li> OpenCV </li>
</ul>

## Datasets
<ul>
<li> HMDB-51</li>
<li> UCF-101 </li>
</ul>
Place the datasets corresponding to path specified in datasets/dataset.py

## Results
### Model
2.4M params, 440M FLOPS, 10MB size 

### Accuracy
<ul>
<li> HMDB-51: 51.3%
<li> UCF-101: 80.6%
</ul>
