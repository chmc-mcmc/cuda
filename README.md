# cuda
5120 parallel Markov chains,  or 10,000,000 samples in three seconds, sample in CUDA and display results in Mathematica

sampler.cu: library

*.cu: driver

*.nb: analysis result

My laptop has cuda 1050:

Multiprocessor count:  5

Max threads per block:  1024

so the maximum parallel threads are 5120.
