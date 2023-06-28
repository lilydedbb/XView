# X-View
  
**X-View: Non-Egocentric Multi-View 3D Object Detector**

**Authors**: Liang Xie, Guodong Xu, Deng Cai, Xiaofei He

\[[`pre-print version`](https://arxiv.org/abs/2103.13001)]

[\[`published version`\]](https://ieeexplore.ieee.org/abstract/document/10049775)

**Introduction**: This repository provides an implementation for our paper "[X-View: Non-Egocentric Multi-View 3D Object Detector](https://arxiv.org/abs/2103.13001)" published on [T-IP](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83). This repository is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). X-View generalizes the research on 3D multi-view learning. Specifically, X-view breaks through the traditional limitation about the perspective view whose original point must be consistent with the 3D Cartesian coordinate to overcome the drawbacks of the multi-view methods.

## Installation

The environment requirements are same as the original code base and you can refer the [instructions](./original_docs/INSTALL.md) to install the environments and prepare the datasets.

## Training & Testing

Please refer the [instructions](./original_docs/GETTING_STARTED.md) in the original code base to perform training and testing. The only thing you should do is changing the path to the configure file.

For example, 

```
sh scripts/dist_train.sh 4 --cfg_file tools/cfgs/kitti_models/second_mv2.yaml
```

## <a></a>Citation

Please consider citing X-View in your publications if it helps your research. :)

```bib
@ARTICLE{10049775,
  author={Xie, Liang and Xu, Guodong and Cai, Deng and He, Xiaofei},
  journal={IEEE Transactions on Image Processing}, 
  title={X-View: Non-Egocentric Multi-View 3D Object Detector}, 
  year={2023},
  volume={32},
  number={},
  pages={1488-1497},
  doi={10.1109/TIP.2023.3245337}
}
```

## Contact

If you have any questions about this work, feel free to contact us through email (Liang Xie: <lilydedbb@gmail.com>) or Github issues.
