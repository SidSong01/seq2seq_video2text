# Tensorflow implementation of seq2seq_v2t, caption generation for videos (video to text)
---

[//]: # (Image References)

[image1]: ./TZ860P4iTaM_15_28.gif

* tensorflow 1.15

* python 3.6

* the training data: https://drive.google.com/open?id=1sSFbOU928jYp1xGx4PF4_hV8_w2kDQ-j

## Details About How to Play

run

```sh
Hw2_seq2seq.sh _ _ _
```

* (_ _ _ here represent "the data directory", "the test data directory", "and the name of the output(.txt)"
the name of the output(.txt) = final_output.txt)

## Results

* the bleu score is about 0.7

* detailed description please refer to the pdf report, thanks.

* sample result for TZ860P4iTaM_15_28.avi

![alt text][image1]

`a cat is playing the piano`

### reference paper: http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf
