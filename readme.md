# Multi-task neural network blind deconvolution and its application to bearing fault feature extraction
This is the offical repository of the paper "Multi-task neural network blind deconvolution and its application to bearing fault feature extraction". Here is the EA vision in Measturement Science and Technology [https://iopscience.iop.org/article/10.1088/1361-6501/accbdb](https://iopscience.iop.org/article/10.1088/1361-6501/accbdb).

In this work,

1. We propose a novel multi-objective optimization function for the BD problem that combines two sparsity criteria: kurtosis and G−l1/l2 norm. Different from previous methods, our innovation lies in the use of these criteria in both time and frequency domains. We also derive the monotonicity of this function, which shows that the opposite monotonicity of the two criteria constrains each other during optimization, further improving the robustness of the BD method.
2. We propose a multi-task one-dimensional convolutional neural network with two branches to achieve the joint optimization of two criteria. To our best knowledge, it is the first time a multi-task CNN has been used for BD problems. Experiments on simulated and real-world bearing fault signals show that our method outperforms other state-of-the-art methods.



All experiments are conducted with Windows 10 on an Intel i9 10900k CPU at 3.70 GHz and one NVIDIA RTX 3080Ti 12GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## Citing
If you find this repo useful for your research, please consider citing it:
```
@article{10.1088/1361-6501/accbdb,
	author={Liao, Jingxiao and Dong, Hangcheng and Luo, Lei and Sun, Jinwei and Zhang, Shiping},
	title={Multi-task neural network blind deconvolution and its application to bearing fault feature extraction},
	journal={Measurement Science and Technology},
	url={http://iopscience.iop.org/article/10.1088/1361-6501/accbdb},
	year={2023},
}
```



## Multi-task Neural Network Blind Deconvolution(MNNBD)

The proposed multi-task neural network blind deconvolution (MNNBD) combines the kurtosis and {$G-l_1/l_2$} norm as a hybrid optimization criterion. The kurtosis is used for time domain signal optimization, and the $G-l_1/l_2$ norm is used for the frequency domain. It complements the shortcomings of these two types of BD methods: i) The time domain-based method (MED [1]), which used the kurtosis as the criterion, may be affected by random pulses without fault characteristic frequency information; ii) The frequency domain-based method (Mini-blp-lplq [2]), which used the $G-l_p/l_q$ norm as the criterion, leads to the high sparsity of the frequency domain but loses the low energy fault-related frequency features. 

![overview](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/overview.png)




## Repository organization

### Requirements
We use PyCharm 2021.2 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.8
* PyTorch == 1.10.1
* CUDA == 11.3 if use GPU
* anaconda == 2021.05
 
### Organization
```
QCNN_for_bearing_diagnosis
│   main.py # MNNBD main entrance
│   Functions.py # two target functions
└─  data # bearing fault datasets 
     │   
     └─ 2HP # example dataset from CWRU
	 └─ sig2.mat # simulated signal
└─  Model
     │   frequency.py # calculate Hilbert transform and get envelope spectrum
     │   NET.py # CNN net
```

### Datasets
We use the CWRU dataset [3] , HIT dataset [4] and XJTU-SY [5] dataset in our article. The CWRU dataset and XJTU-SY dataset are public bearing fault datasets that can be found in [CWRU Dataset](https://github.com/s-whynot/CWRU-dataset) and [XJTU Dataset](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) respectively.

### How to Use
 
Run ```main.py``` to train a MNNBD and get the deconvolution results. The results will be saved to the ***'results'*** folder.
 
### Thanks
We are inspired by these two GitHub repositories for our code. We are  grateful for their open-source contributions.
1. BADBD: https://github.com/FangBo-0219/BADBD
2. Hilbert-Huang Transform: https://github.com/chendaichao/Hilbert-Huang-transform

## Main Results
Here we show the deconvolution result of the simulated signal, which has added -15dB noise. Our method presents an accurate feature extraction and fast speed.

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/results.png)

![enter description here](https://raw.githubusercontent.com/asdvfghg/image/master/QCNN/resultstable.png)


## External source codes
All the baseline methods we utilize their official implement as follows:

 MED [1] \& MCKD [6]: https://www.mathworks.com/matlabcentral/fileexchange/53484-minimum-entropy-deconvolution-multipack-med-meda-omeda-momeda-mckd

SF-SLSN [7]: https://github.com/aresmiki/SF-SLSN

Mini-blp-lplq [2]: https://github.com/aresmiki/Mini-blp-lplq

BADBD [8]: https://github.com/FangBo-0219/BADBD

MNAD [9]: https://github.com/FangBo-0219/MNAD


## Contact
If you have any questions about our work, please contact the following email address:

jingxiaoliao@hit.edu.cn

Enjoy your coding!
## Reference
[1] Ralph A Wiggins. Minimum entropy deconvolution. Geoexploration, 16(1-2):21–35, 1978.

[2] Liu He, Dong Wang, Cai Yi, Qiuyang Zhou, and Jianhui Lin. Extracting cyclo-stationarity of repetitive transients from envelope spectrum based on prior-unknown blind deconvolution technique. Signal Processing, 183:107997, 2021.

[3] https://csegroups.case.edu/bearingdatacenter/pages/download-data-file

[4] Liao, J. X., Dong, H. C., Sun, Z. Q., Sun, J., Zhang, S., & Fan, F. L. (2023). Attention-embedded quadratic network (qttention) for effective and interpretable bearing fault diagnosis. IEEE Transactions on Instrumentation and Measurement.

[5] Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, “A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings”, IEEE Transactions on Reliability, vol. 69, no. 1, pp. 401-412, 2020. DOI: 10.1109/TR.2018.2882682.

[6] McDonald, G. L., Zhao, Q., & Zuo, M. J. (2012). Maximum correlated Kurtosis deconvolution and application on gear tooth chip fault detection. Mechanical Systems and Signal Processing, 33, 237-255.

[7] He, Liu, et al. “Optimized Minimum Generalized Lp/Lq Deconvolution for Recovering Repetitive Impacts from a Vibration Mixture.” Measurement, vol. 168, Elsevier BV, Jan. 2021, p. 108329, doi:10.1016/j.measurement.2020.108329.

[8] Fang, B., Hu, J., Yang, C., Cao, Y., & Jia, M. (2021). A blind deconvolution algorithm based on backward automatic differentiation and its application to rolling bearing fault diagnosis. Measurement Science and Technology, 33(2), 025009.

[9] Fang, B., Hu, J., Yang, C., & Chen, X. (2022). Minimum noise amplitude deconvolution and its application in repetitive impact detection. Structural Health Monitoring, 14759217221114527.
