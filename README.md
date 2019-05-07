# OverPainter
This is a improved version of Deepcolor
![pic_failed](http://github.com/APTXOUS/OverPainter/MD/demo.png)
# Environment
    tensorflow 1.13
    CUDA Version 10.1
    Ubuntu 18.04 LTS
    python 2.7
# How to use it
    cd Overpainter
    mkdir test       #for store test result
    mkdir imgs       #for train
    mkdir imgs-vaild #for test
Place your training data in the imgs holder,and run   
    
    python Overpainter train

We use 10780 pictures of 512*512 size,training for about 48 hours.   
To sample

    python Overpainter sample


