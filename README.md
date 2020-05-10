# UFC: Unrestricted Facial reConstruction

Implementation of "Unrestricted Facial Geometry Reconstruction Using Image-to-Image Translation" by Sela et al.

### Requirements:

- torch>=0.4.1
- torchvision>=0.2.1
- dominate>=2.3.1
- visdom>=0.1.8.3
- MATLAB

### Dataset and Weights

https://drive.google.com/drive/folders/1QCvw73mISKDoT2Alpv0FAPbEC2U3I_mL?usp=sharing

### Usage

Download `rgb2depth_dataset.zip` and `depth_generator_network.pth` from the above google drive link.

For Depth Estimation

```
git clone https://github.com/vikasTmz/ufc.git;
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git;

unzip rgb2depth_dataset;
mv rgb2depth_dataset pytorch-CycleGAN-and-pix2pix/datasets;

mv depth_generator_network.pth latest_net_G.pth;
mkdir pytorch-CycleGAN-and-pix2pix/checkpoints/rgb2depth_pix2pix;
mv latest_net_G.pth pytorch-CycleGAN-and-pix2pix/checkpoints/rgb2depth_pix2pix;

cd pytorch-CycleGAN-and-pix2pix;
python test.py --dataroot ./datasets/rgb2depth_dataset --name rgb2depth_pix2pix --model pix2pix --direction AtoB;
```

For Geometric Reconstruction:
```
cd ufc/src;
python demo.py --rgb_img <path/to/rgb/image> --depth_img <path/to/depth/image> --correspondence_img <path/to/correspondence/image> --output_name output
```


