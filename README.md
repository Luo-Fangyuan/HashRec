# <p align="center">Learning to Hash for Recommendation</p>

This website collects recent works and datasets on learning to hash for recommendation and their codes. We hope this website could help you do search on this topic.

##### If you find our project is useful to your research or work, please give us a star ⭐ on GitHub for the latest update and cite our paper. Thank you!

> **Paper**: [Learning to Hash for Recommendation: A Survey](https://arxiv.org/abs/2412.03875)

> **Authors**: Fangyuan Luo<sup>1</sup>, Honglei Zhang<sup>2</sup>, Tong Li<sup>1</sup>, Jun Wu<sup>2</sup>, Guandong Xu<sup>3</sup>, Haoxuan Li<sup>4</sup>

> **Affliation**: <sup>1</sup>Beijing University of Technology, <sup>2</sup>Beijing Jiaotong University, <sup>3</sup>The Education University of Hong Kong, <sup>4</sup>Peking University


## Surveys

1. A. Singh and S. Gupta, “[Learning to hash: a comprehensive survey of deep learning-based hashing methods](https://link.springer.com/article/10.1007/s10115-022-01734-0),” *Knowledge and Information Systems*, vol. 64, no. 10, pp. 2565–2597, 2022.
2. J. Wang, W. Liu, S. Kumar, and S. Chang, “[Learning to hash for indexing big data - A survey](https://ieeexplore.ieee.org/document/7360966),” *Proceedings of the IEEE*, vol. 104, no. 1, pp. 34–57, 2016.
3. J. Wang, T. Zhang, J. Song, N. Sebe, and H. T. Shen, “[A survey on learning to hash](https://ieeexplore.ieee.org/abstract/document/7915742),” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 4, pp. 769–790, 2018.
4. X.Luo,H.Wang,D.Wu,C.Chen,M.Deng,J.Huang,andX.Hua,“[A survey on deep hashing methods](https://dl.acm.org/doi/full/10.1145/3532624),” *ACM Transactions on Knowledge Discovery from Data*, vol. 17, no. 1, pp. 15:1–15:50, 2023.
5. Z. Li, H. Li, and L. Meng, “[Model compression for deep neural networks: A survey](https://www.mdpi.com/2073-431X/12/3/60),” *Computers*, vol. 12, no. 3, p. 60, 2023.

## Related Papers

|                       Paper                       |                       Model                       | Venue | Task | Learning Objective | Optimization Strategy |                           PDF                           |    Code    |
| :----------------: | :--------: | :--------: | ------------------------------------------------- | ------------------------------------------------- | :-----------------------------------------------: | ------------------------------------------------- | ------------------------------------------------- |
| Learning binary codes for collaborative filtering | BCCF |KDD'12|User-Item CF|Pointwise, Pairwise|Two-Stage| [PDF](https://dl.acm.org/doi/10.1145/2339530.2339611)| [Code](https://github.com/DefuLian/recsys/tree/master/alg/discrete)|
| Collaborative Hashing | CH |CVPR'14|User-Item CF|Pointwise|Two-Stage| [PDF](https://xlliu-beihang.github.io/file/cvpr2014.pdf)| [Code](https://github.com/DefuLian/recsys/tree/master/alg/discrete)|
| Preference preserving hashing for efficient recommendation | PPH |SIGIR'14|User-Item CF|Pointwise|Two-Stage| [PDF](https://dl.acm.org/doi/10.1145/2600428.2609578)| [Code](https://github.com/DefuLian/recsys/tree/master/alg/discrete)|
| Discrete Collaborative Filtering | DCF |SIGIR'14|User-Item CF|Pointwise|One-Stage| [PDF](http://staff.ustc.edu.cn/~hexn/papers/sigir16-dcf-cm.pdf)| [Code](https://github.com/hanwangzhang/Discrete-Collaborative-Filtering)|
| Discrete Content-aware Matrix Factorization | DCMF |KDD'17|Cold-Start|Pointwise|One-Stage| [PDF](https://dl.acm.org/doi/10.1145/3097983.3098008)| [Code](https://github.com/DefuLian/recsys/tree/master/alg/discrete/dcmf)|
| Discrete Personalized Ranking for Fast Collaborative Filtering from Implicit Feedback | DPR |AAAI'17|User-Item CF|Pairwise|One-Stage| [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/10764)| [Code](https://github.com/yixianqianzy/dpr)|
| Discrete Factorization Machines for Fast Feature-based Recommendation | DFM |IJCAI'18|Cold-Start|Pointwise|One-Stage| [PDF](https://www.ijcai.org/Proceedings/2018/0479.pdf) | [Code](https://github.com/hanliu95/DFM) |
| Learning Discrete Hashing Towards Efficient Fashion Recommendation | DSFCH |Data Science and Engineering'18|Outfit Recommendation|Pointwise|One-Stage| [PDF](https://link.springer.com/article/10.1007/s41019-018-0079-z)| [Code]()|
| Discrete Ranking-based Matrix Factorization with Self-Paced Learning | DRMF |KDD'18|User-Item CF|Pointwise|One-Stage| [PDF](https://bigdata.ustc.edu.cn/paper_pdf/2018/Yan-Zhang-KDD2018.pdf)| [Code](https://github.com/yixianqianzy/drmf-spl)|
| Discrete Deep Learning for Fast Content-Aware Recommendation | DDL |WSDM'18|Cold-Start|Pointwise|One-Stage| [PDF](https://dl.acm.org/doi/10.1145/3159652.3159688)| [Code](https://github.com/yixianqianzy/ddl)|
| Discrete Trust-aware Matrix Factorization for Fast Recommendation | DTMF |IJCAI'19|Social Recommendation|Pointwise|One-Stage| [PDF](https://www.ijcai.org/proceedings/2019/0191.pdf)| [Code](https://github.com/EnnengYang/DTMF)|
| Candidate Generation with Binary Codes for Large-Scale Top-N Recommendation | CIGAR |CIKM'19|User-Item CF|Pairwise|Two-Stage| [PDF](https://dl.acm.org/doi/pdf/10.1145/3357384.3357930)| [Code](https://github.com/kang205/CIGAR)|
| Compositional Coding for Collaborative Filtering | CCCF |SIGIR'19|User-Item CF|Pointwise|One-Stage| [PDF](https://dl.acm.org/doi/10.1145/3331184.3331206)| [Code](https://github.com/3140102441/CCCF)|
| Discrete Social Recommendation | DSR |AAAI'19|Social Recommendation|Pointwise|One-Stage| [PDF](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.3301208#:~:text=Social%20recommendation%2C%20which%20aims%20at,represent%20user%2Fitem%20latent%20features.)| [Code](https://github.com/3140102441/Discrete-Social-Recommendation)|
| Learning Binary Code for Personalized Fashion Recommendation | FHN |CVPR'19|Outfit Recommendation|Pairwise|Two-Stage| [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Learning_Binary_Code_for_Personalized_Fashion_Recommendation_CVPR_2019_paper.pdf)| [Code](https://github.com/lzcn/Fashion-Hash-Net)|
| Binarized Collaborative Filtering with Distilling Graph Convolutional Networks | DGCN-BinCF |IJCAI'19|User-Item CF|Pairwise|Two-Stage| [PDF](https://www.ijcai.org/proceedings/2019/0667.pdf)| [Code]()|
| Adversarial Binary Collaborative Filtering for Implicit Feedback | ABinCF |AAAI'19|User-Item CF|Pointwise|Two-Stage| [PDF](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33015248)| [Code]()|
| Content-aware Neural Hashing for Cold-start Recommendation | NeuHash-CF |SIGIR'20|Cold-Start|Pointwise|Two-Stage| [PDF](https://dl.acm.org/doi/abs/10.1145/3397271.3401060)| [Code](https://github.com/casperhansen/NeuHash-CF)|
| Learning to Hash with Graph Neural Networks for Recommender Systems | HashGNN |WWW'20|User-Item CF|Heterogeneous|Two-Stage| [PDF](https://dl.acm.org/doi/10.1145/3366423.3380266)| [Code](https://github.com/qiaoyu-tan/HashGNN)|
| Semi-discrete Matrix Factorization | SDMF |IEEE Intelligent Systems'20|User-Item CF|Pointwise|One-Stage| [PDF](https://ieeexplore.ieee.org/document/9171422)| [Code](https://github.com/Luo-Fangyuan/SDMF)|
| Multi-Feature Discrete Collaborative Filtering for Fast Cold-start Recommendation | MFDCF |AAAI'20|Cold-Start|Pointwise|One-Stage| [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/5360)| [Code]()|
| Collaborative Generative Hashing for Marketing and Fast Cold-Start Recommendation | CGH |IEEE Intelligent Systems'20|Cold-Start|Pointwise|Two-Stage|[PDF](https://ieeexplore.ieee.org/document/9200727)| [Code](https://github.com/yixianqianzy/cgh)|
| Deep Pairwise Hashing for Cold-start Recommendation | DPH |IEEE TKDE'22|Cold-Start|Pairwise|One-Stage| [PDF](https://ieeexplore.ieee.org/document/9197722)| [Code](https://github.com/yixianqianzy/dph)|
| Projected Hamming Dissimilarity for Bit-Level Importance Coding in Collaborative Filtering | VH_{PHD} |WWW'21|User-Item CF|Pointwise|Two-Stage| [PDF](https://dl.acm.org/doi/abs/10.1145/3442381.3450011#:~:text=Projected%20Hamming%20Dissimilarity%20for%20Bit%2DLevel%20Importance%20Coding%20in%20Collaborative%20Filtering,-Christian%20Hansen%2C%20University&text=When%20reasoning%20about%20tasks%20that,be%20done%20efficiently%20and%20effectively.)| [Code](https://github.com/casperhansen/Projected-Hamming-Dissimilarity)|
| Discrete Matrix Factorization and Extension for Fast Item Recommendation | DMF |IEEE TKDE'21|Cold-Start|Pointwise|One-Stage| [PDF](https://ieeexplore.ieee.org/document/8890891)| [Code](https://github.com/DefuLian/recsys2)|
| Discrete Listwise Collaborative Filtering for Fast Recommendation | DLCF |SDM'21|User-Item CF|Listwise|Proximal One-Stage| [PDF](https://epubs.siam.org/doi/10.1137/1.9781611976700.6)| [Code]()|
| Semi-Discrete Social Recommendation (Student Abstract) | SDSR |AAAI'21|Social Recommendation|Pointwise|One-Stage| [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/17914)| [Code]()|
| Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation | BiGeaR |KDD'22|User-Item CF|Pairwise|Two-Stage| [PDF](https://dl.acm.org/doi/10.1145/3534678.3539452)| [Code](https://github.com/yankai-chen/BiGeaR)|
| Bi-directional Heterogeneous Graph Hashing towards Efficient Outfit Recommendation | BIHGH |ACM MM'22|Outfit Recommendation|Pairwise|Two-Stage| [PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3548020)| [Code](https://github.com/Hyu-Zhang/BiHGH)|
| Discrete Listwise Personalized Ranking for Fast Top-N Recommendation with Implicit Feedback | DLPR |IJCAI'22|User-Item CF|Listwise|One-Stage| [PDF](https://www.ijcai.org/proceedings/2022/0300.pdf)| [Code](https://github.com/Luo-Fangyuan/DLPR)|
| HCFRec Hash Collaborative Filtering via Normalized Flow with Structural Consensus for Efficient Recommendation | HCFRec |IJCAI'22|User-Item CF|Pointwise|Two-Stage| [PDF](https://www.ijcai.org/proceedings/2022/0315.pdf)| [Code]()|
| Explainable discrete Collaborative Filtering | EDCF |IEEE TKDE'22|Explainable Recommendation|Pointwise|Two-Stage| [PDF](https://ieeexplore.ieee.org/document/9802915)| [Code](https://github.com/zzmylq/EDCF)|
| Discrete Limited Attentional Collaborative Filtering for Fast Social Recommendation | DLACF |EAAI'23|Social Recommendation|Pointwise|Two-Stage| [PDF](https://www.sciencedirect.com/science/article/pii/S0952197623006218) | [Code](https://github.com/qhgz2013/DLACF) |
| Personalized Fashion Recommendation With Discrete Content-Based Tensor Factorization | FHN+ |IEEE TMM'23|Outfit Recommendation|Pairwise|Two-Stage| [PDF](https://ieeexplore.ieee.org/document/9808340) | [Code]() |
| Bipartite Graph Convolutional Hashing for Effective and Efficient Top-N Search in Hamming Space | BGCH |WWW'23|User-Item CF|Heterogeneous|Two-Stage| [PDF](https://dl.acm.org/doi/abs/10.1145/3543507.3583219)| [Code](https://github.com/yankai-chen/BGCH)|
| Multi-Modal Discrete Collaborative Filtering for Efficient Cold-Start Recommendation | MDCF |IEEE TKDE'23|Cold-Start|Pointwise|One-Stage| [PDF](https://ieeexplore.ieee.org/document/9429954)| [Code](https://github.com/zzmylq/MDCF)|
| LightFR: Lightweight Federated Recommendation with Privacy-preserving Matrix Factorization | LightFR | ACM TOIS'23 | Federated Recommendation | Pointwise | One-Stage | [PDF](https://ieeexplore.ieee.org/document/9429954)| [Code](https://github.com/hongleizhang/LightFR)|
| Discrete Listwise Content-aware Recommendation | DLFM | ACM TKDD'24 | Cold-Start | Listwise | Proximal One-Stage | [PDF](https://dl.acm.org/doi/10.1145/3609334) | [Code]() |
| Discrete Federated Multi-behavior Recommendation for Privacy-Preserving Heterogeneous One-Class Collaborative Filtering | DFMR | ACM TOIS'24 | Federated Recommendation | Pointwise | One-Stage | [PDF](https://dl.acm.org/doi/10.1145/3652853) | [Code]() |
| Temporal Social Graph Network Hashing for Efficient Recommendation | TSGNH | IEEE TKDE'24 | Social Recommendation | Pointwise | Two-Stage | [PDF](https://ieeexplore.ieee.org/document/10387583)| [Code](https://github.com/zzmylq/TSGNH)|
| Towards Effective Top-N Hamming Search via Bipartite Graph Contrastive Hashing | BGCH+ | IEEE TKDE'24 | User-Item CF | Pairwise | Two-Stage | [PDF](https://ieeexplore.ieee.org/document/10638796)| [Code]()|

## Metrics

We contain the implementation of evaluation metrics for recommender systems. Suppose that there are two users. The label and prediction list are ```[[1,1,1,1,0,0,0,0,1,1,1,1], [1,1,0,0,1]]``` and ```[[1,1,0,1,0,0,0,0,0,0,1,0], [0,1,1,0,0]]``` respectively, where each sublist in label or prediction corresponds to one user's label list or prediction list. And ```K``` denotes the cut-off of the list.

```python
from RecMetrics import Metric

metric = Metric(label,pred, K)

hitratio = metric.hit_ratio()
recall = metric.recall()
accuracy = metric.accuracy()
mae = metric.mae()
rmse = metric.rmse()
ndcg = metric.ndcg()
map = metric.map()
auc = metric.auc()
mrr = metric.mrr()
```

## Datasets

We collect some datasets ([Download](https://drive.google.com/drive/folders/162l1Xxme5ic0vN5Q8VsfrJZMyPBtCafI)) which are often used in the research of HashRec.

|                       Datasets                    | #Users             |                           #Items                        |   #Interactions    | Density |
| :-----------------------------------------------: | :----------------: | :-----------------------------------------------------: | :----------------: |:----------------: |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,952 | 1,000,209 |4.19% |
| [Movielens 10M](https://grouplens.org/datasets/movielens/10m/) | 71,567 | 10,681 | 10,000,054| 1.31% |
| [EachMovie](https://networkrepository.com/rec-eachmovie.php) | 1,648 | 74,424 | 2,811,717 | 2.83% |
| [Netflix](https://tianchi.aliyun.com/dataset/146311) | 480,189|17,770|100,480,507| 1.18%|
|[Yelp](https://www.yelp.com/dataset) | 13,679|12,922|640,143|0.36%|
|[Amazon Book](http://jmcauley.ucsd.edu/data/amazon/) | 35,151|33,195|1,732,060|0.15%|
| [Gowalla](https://drive.google.com/file/d/0BzpKyxX1dqTYRTFVYTd1UG81ZXc/view?pli=1&resourcekey=0-SeoSMLHJTnTRO-SRJN_xHA) | 29,858 | 40,981 | 1,027,370| 0.08%|
|[Pinterest](https://data.mendeley.com/datasets/fs4k2zc5j5/3) | 55,186|9,916|1,463,556| 0.27%|
| [BookCrossing](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset/data) | 278,858 |271,379 | 1,149,780 | 0.002% |

## Tips

🔥🔥 We will keep updating this list, and if you find any missing related work or have any suggestions, please feel free to contact us (luofangyuan@bjut.edu.cn).

```
@article{Luo2024HashRec,
  title = {Learning to Hash for Recommendation: A Survey},
  author = {Fangyuan Luo, Honglei Zhang, Tong Li and Jun Wu},
  year = {2024},
  journal = {arXiv preprint arXiv: 2412.03875}
}
```
