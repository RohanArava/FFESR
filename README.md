# Pytorch implementation of Feature Fusion Enhanced Super Resolution for Low Bitrate Screen Content Compression

The dataset used is present in [https://drive.google.com/drive/folders/1uTQ2FAAUz5l-rtP35_fUhByRXxO25IFW](https://drive.google.com/drive/folders/1uTQ2FAAUz5l-rtP35_fUhByRXxO25IFW)

Download and place the SCI1K-Train, SCI1k-Test in the data folder.  
Modify Args.py file too represent the paths to the data folders and result folders.  

Install required libraries through:
```
pip install -r requirements.txt
```
> Note: Make sure you have a CUDA GPU avalilable

Run the project using:
```
python main.py
```

The models will be saved every epoch in `save_path/Models` folder.  
The plots will be saved in `save_path/Plots` folder.  
The plot shows ssim scores vs compression ratio. Compare that against the respective plots in the paper. [[1]](#1)

## References

<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10222724&isnumber=10221892" id="1">[1]</a>  
T. Tang, X. Zhang, Z. Li and J. Yang, "Feature Fusion Enhanced Super Resolution for Low Bitrate Screen Content Compression," 2023 IEEE International Conference on Image Processing (ICIP), Kuala Lumpur, Malaysia, 2023, pp. 2825-2829, doi: 10.1109/ICIP49359.2023.10222724.  
Abstract: Screen content image/video has been widely applied into cloud game, AR/VR, online education etc., while the transmission and storage of screen content data is limited, and existing compression methods tend to lose details when encoding screen content with low bitrate. To address this challenge, a low bitrate screen content compression method based on super-resolution is proposed in this paper. The method combines super-resolution (SR) and down-sampling techniques to compress screen content while preserving its details. A novel super-resolution network is designed, which enhances detail preservation during feature fusion and eliminates compression distortion artifacts. The proposed method was evaluated against the state-of-the-art HEVC-SCC and can save bit rate by 19.77% and 26.18% on the SCID and SIQAD datasets, respectively while maintaining similar subjective quality.  
keywords: {Image coding;Superresolution;Bit rate;Education;Rate-distortion;Games;Distortion;Screen content coding;super resolution;high efficiency video coding (HEVC)},  
URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10222724&isnumber=10221892