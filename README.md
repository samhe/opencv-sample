# 0基础用OpenCV做人脸识别示例

## Environment

* Mac Book Air
* Python3
* OpenCV2

## Folder Structure

```
├── cv-camera.py       #用opencv动态识别mac摄像头的数据，并在适当时候截图
├── cv-image.py        #用opencv识别一个图片的人脸
├── data               #从opencv上复制过来的特征数数
├── images
│   └── sample.jpeg    #cv-image.py 的默认识别样图片
└── result             #ca-camera.py 的输出目录
```


## Installations (for Mac only)

* Ensure brew, xcode installed and update
* Install the opencv by brew (最好翻墙后搞，会顺利点。)
    ```shell
    $ brew install opencv
    Updating Homebrew...
    ...
    ...
    ...
    ```
* Verification
    ```shell
    $ python3
    Python 3.6.4 (default, Jan  6 2018, 11:51:15)
    [GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import cv2
    ```
    > 按理Brew应该会给python2和python3都安装opencv的库，但不知为什么我的mac安装后，只有python3管用。有经验的同学，麻烦指导一下
* Install the additional libs by pip
    ```shell
    pip3 install -r requirements.txt
    ```

## Trail run

* cv-image.py - detect the face from a image and draw the result in a new python window
    ```shell
    python3 cv-image.py

    # or you can specify a image you want
    python3 cv-image.py --image images/sample.jpeg

    ```

    - you can press any key to exits

* cv-camera.py
    ```shell
    python3 cv-camera.py

    # this script will load the source from camera and detect the face real time
    ```

    - press 'c' to capture a image
    - press 'q' to exits
    - the program will capture max. 50 screen in the result folder, when it got new face detected or 'c' key press
    - when the program exists, it will detec the face of the screen captured in result folder and save the related faces in subfolder.


