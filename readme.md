# Tensorflow implementation of seq2seq_v2t

## Video caption generation (video to text)

This is the second assignment of CPSC 8810, Clemson Palemtto is used.

[//]: # (Image References)

[image1]: ./TZ860P4iTaM_15_28.gif
[image2]: ./Annotation%202020-05-27%20014911.png

### Requirements
* tensorflow 1.15

* python 3.6

* the training data: https://drive.google.com/open?id=1sSFbOU928jYp1xGx4PF4_hV8_w2kDQ-j

### Model architecture

![alt text][image2]

### Details About How to Play

run

* For running the testing program and then evaluate it wiht bleu score:
```sh
Hw2_seq2seq.sh _ _ _
```

#### * (_ _ _ here represent "the data directory", "the test data directory", and "the name of the output(.txt)", the name of the output(.txt) = final_output.txt)

```sh
bleu_eval.py
```
for output bleu score.


### Results

* the bleu score is about 0.7, there is lots of ways to better it, such as making use of attention model, ...

* detailed description please refer to the pdf report, thanks.

* sample result for TZ860P4iTaM_15_28.avi

![alt text][image1]

`a cat is playing the piano`

#### Reference paper: http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf
