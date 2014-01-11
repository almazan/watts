Word Spotting and Recognition with Embedded Attributes
==

Welcome to the Word Representation with Attributes library, a software for the retrieval and recognition of word images.

This code is written in Matlab and is the basis of the following project:

###### [Jon Almazán](http://www.cvc.uab.es/~almazan), Albert Gordo, Alicia Fornés, Ernest Valveny.  *Word Spotting and Recognition with Embedded Attributes*. [Project Page](http://www.cvc.uab.es/~almazan/index.php/projects/words-att/)

![](http://www.cvc.uab.es/~almazan/wp-content/uploads/2012/07/spaces.png)

Abstract
---

We deal with the problems of word spotting and word recognition on images. In word spotting, the goal is to find all instances of a query word in a dataset of images. In recognition, the goal is to recognize the content of the word image, usually aided by a dictionary or lexicon. We propose a formulation for word representation and matching based on embedded attributes that jointly addresses these two problems. Contrary to most other existing methods, our representation has a fixed length, is low dimensional, and is very fast to compute and, especially, to compare.

We propose to use character attributes to learn a semantic representation of the word images and then perform a calibration of the scores with CCA that puts images and text strings in a common subspace. After that, spotting and recognition become simple nearest neighbor problems in a very low dimensional space. We test our approach on four public datasets of both document and natural images showing results comparable or better than the state-of-the-art on spotting and recognition tasks.

This word spotting library uses some great open-source software:

* [Yael library](https://gforge.inria.fr/projects/yael/) 
* [VLFeat](http://www.vlfeat.org)

----

# MATLAB Quick Start Guide

To get started, you need to install MATLAB and download the code from Github. This code has been tested on Mac and Linux and pre-compiled Mex files are included.

## Download source code
``` sh
$ cd ~/your_projects/
$ git clone git://github.com/almazan/words-att.git
```

## Download and uncompress datasets
``` sh
$ cd ews/datasets
$ wget http://www.cvc.uab.es/~almazan/data/Datasets.tar.gz
$ tar -xzf Datasets.tar.gz
```

## Script for parameters validation

``` sh
>> validation_script
```
