# Predict tennis ball landing point in real environment

## Development environment:

  1. OS : Ubunut 20.04
  2. CUDA : 11.4
  3. GPU : GTX 1660ti

## 1. Setup:

```bash
conda create -n tennis_project python=3.7
conda env update -f environment.yaml

cd src_yolov5/src
```


## 2. Run Demo:

* Start predict tennis ball landing point

```bash
python main.py --video_path=your_video_path
```

<p align="center">
<img width="100%" src="https://user-images.githubusercontent.com/67572161/152946889-32039eeb-b8b5-4d9c-b25c-5db22a62e438.gif"/>
  
</p>
