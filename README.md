Word Spotting and Recognition with Embedded Attributes
==

Welcome to the Word Representation with Attributes library, a software for the retrieval and recognition of word images.

This code is written in Matlab and is the basis of the following project:

###### [Jon Almazán](http://www.cvc.uab.es/~almazan), [Albert Gordo](http://dag.cvc.uab.es/content/albert-gordo), [Alicia Fornés](http://dag.cvc.uab.es/content/alicia-forn%C3%A9s), [Ernest Valveny](http://www.cvc.uab.es/personal2.asp?id=73).  *Word Spotting and Recognition with Embedded Attributes*. [Project Page](http://www.cvc.uab.es/~almazan/index.php/projects/words-att/)

![](http://www.cvc.uab.es/~almazan/wp-content/uploads/2014/01/spaces_small.png)

Abstract
---

We deal with the problems of word spotting and word recognition on images. In word spotting, the goal is to find all instances of a query word in a dataset of images. In recognition, the goal is to recognize the content of the word image, usually aided by a dictionary or lexicon. We propose a formulation for word representation and matching based on embedded attributes that jointly addresses these two problems. Contrary to most other existing methods, our representation has a fixed length, is low dimensional, and is very fast to compute and, especially, to compare.

We propose to use character attributes to learn a semantic representation of the word images and then perform a calibration of the scores with CCA that puts images and text strings in a common subspace. After that, spotting and recognition become simple nearest neighbor problems in a very low dimensional space. We test our approach on four public datasets of both document and natural images showing results comparable or better than the state-of-the-art on spotting and recognition tasks.

This word spotting library uses some great open-source software:

* [Yael library](https://gforge.inria.fr/projects/yael/) 
* [VLFeat](http://www.vlfeat.org)

----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from GitHub. This code has been tested on Mac and Linux and some pre-compiled Mex files are included.

## Download source code
``` sh
$ cd ~/your_projects/
$ git clone git://github.com/almazan/words-att.git
```

## Download and uncompress the IIIT5K dataset
``` sh
$ cd words-att/datasets
$ wget http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
$ tar -xvzf IIIT5K-Word_V3.0.tar.gz
```

## Run the program with the default parameters

``` sh
>> main
```

Note: The default parameters as well as the dataset selection can be modified in the *prepare_opts.m* script.

----

# Authors

The code has been developed by @almazan and @agordo.

# License

This code has been released under a GPL license. If you need a version of the code licensed under MIT, please, [contact me](http://www.cvc.uab.es/~almazan).
