# cuda_and_contours
Classic computer vision operations using NVIDIA GPUs

Using only computer vision techniques (edge detection, thresholding, blurring, etc.) a developer can put together useful applications without having to use computational expensive machine learning algorithms. For example using background subtraction a developer could create an application that counts people going in and out of a store, or an application that counts cars entering and exiting a parking garage, or a motion detection application for use in security cameras. If developers wish to track the objects detected from background subtraction they simply apply contours to them. To speed up these applications a developer can use CUDA APIs to offload computer vision algorithms from the CPU to the GPU.

## Repo Programs
| Folder                     	| Description                                                                                              	|
|----------------------------	|----------------------------------------------------------------------------------------------------------	|
| background_subtraction   | Program use MOG2 background subtractor to track people in a video frame using either CUDA or CPU to execute the algorithm|
| color_space_cuda 	       | Program demonstrate how to optimally use CUDA in your application to move from one color space to another and simultaneously display them to the screen in real time.|

## Setup

This app requires an alwaysAI account.  Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI tools on your development machine.

## Usage

Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can run the following CLI commands:

To set up the target device & install path

```
aai app configure
```

To install the app to your target

```
aai app install
```

To start the app

```
aai app start
```
