# KWYAF

**Official PyTorch implementation of**  
📄 *[Know Where You Are From: Event-Based Segmentation via Spatio-Temporal Propagation](https://ojs.aaai.org/index.php/AAAI/article/view/32508)*  
Accepted at AAAI 2025 🎉

---

### 🔧 Overview

This repository contains the official PyTorch implementation of the **KWYAF** framework for event-based semantic segmentation.  

🗓️ **Code and full documentation will be released in August or September 2025**. Stay tuned!

If you have any questions or feedback, feel free to open an issue or contact us via email. 😊

---

### 📌 Update (Aug 17, 2025)

- Added the core **dataset** and **model** modules.  
  - `dst/sequence` can be used following the reference implementation from [DSEC Sequence Dataset](https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/sequence.py).  
  - `model/segformer_build` includes model configuration settings. You may replace the default **MiT-B0 backbone** (used in the paper) with stronger variants. The `EncoderDecoder` class handles the feature processing pipeline.  

⚠️ Please note: the authors are currently occupied with job applications and other work. A complete one-click reproduction of the results will be released later.  
In the meantime, if you encounter any issues during reproduction, feel free to reach out for discussion!  



---
### License
This project is only for academic use.

### Acknowledgement
The code is heavily based on the following repositories:
- https://github.com/open-mmlab/mmsegmentation
- https://github.com/NVlabs/SegFormer
- https://github.com/GuoleiSun/VSS-CFFM
- https://github.com/uzh-rpg/DSEC

Thanks for their amazing works.
### 📌 Citation

If you find our work helpful, please consider citing us:

```bibtex
@inproceedings{li2025know,
  title={Know Where You Are From: Event-Based Segmentation via Spatio-Temporal Propagation},
  author={Li, Ke and Lyu, Gengyu and Chen, Hao and Xie, Bochen and Yang, Zhen and Li, Youfu and Deng, Yongjian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={4806--4814},
  year={2025}
}
```
### Contact
- Ke Li, toKeLi@emails.bjut.edu.cn
