# KWYAF

**Official PyTorch implementation of**  
📄 *[Know Where You Are From: Event-Based Segmentation via Spatio-Temporal Propagation](https://ojs.aaai.org/index.php/AAAI/article/view/32508)*  
Accepted at AAAI 2025 🎉

---

### 🔧 Overview

This repository contains the official PyTorch implementation of the **KWYAF** framework for event-based semantic segmentation.  


🗓️ **Code and full documentation will be released in May or June 2025**. Stay tuned!

If you have any questions or feedback, feel free to open an issue or contact us via email. 😊

---

### 📌 Citation

If you find our work helpful, please consider citing us:

```bibtex
@article{Li_Lyu_Chen_Xie_Yang_Li_Deng_2025,
  title     = {Know Where You Are From: Event-Based Segmentation via Spatio-Temporal Propagation},
  author    = {Li, Ke and Lyu, Gengyu and Chen, Hao and Xie, Bochen and Yang, Zhen and Li, Youfu and Deng, Yongjian},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {5},
  pages     = {4806--4814},
  year      = {2025},
  month     = {Apr.},
  doi       = {10.1609/aaai.v39i5.32508},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/32508},
  abstract  = {
    Event cameras have gained attention in segmentation due to their higher temporal resolution and dynamic range compared to traditional cameras. However, they struggle with issues like lack of color perception and triggering only at motion edges, making it hard to distinguish objects with similar contours or segment spatially continuous objects. Our work aims to address these often overlooked issues. Based on the assumption that various objects exhibit different motion patterns, we believe that embedding the historical motion states of objects into segmented scenes can effectively address these challenges. Inspired by this, we propose the ESS framework ``Know Where You Are From" (KWYAF), which incorporates past motion cues through spatio-temporal propagation embedding. This framework features two core components: the Sequential Motion Encoding Module (SME) and the Event-Based Reliable Region Selection Mechanism (ER²SM). SMEs construct prior motion features through spatio-temporal correlation modeling for boosting final segmentation, while ER²SM adapts to identify high-confidence regions, embedding motion more precisely through local window masks and reliable region selection. A large number of experiments have demonstrated the effectiveness of our proposed framework in terms of both quantity and quality.
  }
}
