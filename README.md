# Multi-Fish-Tracking-with-Underwater-Image-Enhancement-by-Deep-Network-in Marine-Ecosystems
Tracking of marine life is an important part of the study and analysis of migration patterns, movements, and population growth of underwater animals. Fish-tracking deep learning networks have been under research and development for quite some time and have been able to produce satisfactory results. We propose an end-to-end deep learning framework to track fishes in unconstrained marine environment. The novel and a key component of this network is image enhancement based Siamese architecture that measures appearance similarity. The image enhancement is performed by the proposed enhancement module, which consists of convolutional layers along with a squeeze and excitation block. The module is pre-trained using degraded and clean image pairs to deal with the underwater degradation. The enhanced feature is utilized by the Siamese framework to produce an appearance similarity score, which is further complemented by prediction scores that consider the fish movement pattern. The appearance similarity score, prediction score, and IOU-based similarity scores are combined to produce fish trajectories using the Hungarian algorithm. This framework is capable of reducing the number of ID switches by a margin of 35.6\% on Fish4Knowledge dataset and 3.8\% on GMOT-40 fish category while maintaining high tracking accuracy.

![arch_upm](https://github.com/user-attachments/assets/72efc45c-b4fa-4d4a-b748-3f37faeabbab)


![enhance](https://github.com/user-attachments/assets/d8c5bce6-3e13-44b4-ad76-d27a3baea92c)

This repository is an implementation of the paper titled "Multi-fish tracking with underwater image enhancement by deep network in marine ecosystems" from Signal Processing: Image Communication journal. If you find our work useful in your work, please cite our work:
Prerana Mukherjee, Srimanta Mandal, Koteswar Rao Jerripothula, Vrishabhdhwaj Maharshi, Kashish Katara, “Multi-Fish Tracking with Underwater Image Enhancement by Deep Network in Marine Ecosystems,” in Signal Processing: Image Communication, Vol.138, pp.117321, 2025.

```
@article{MUKHERJEE2025117321,
title = {Multi-fish tracking with underwater image enhancement by deep network in marine ecosystems},
journal = {Signal Processing: Image Communication},
volume = {138},
pages = {117321},
year = {2025},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2025.117321},
url = {https://www.sciencedirect.com/science/article/pii/S0923596525000682},
author = {Prerana Mukherjee and Srimanta Mandal and Koteswar Rao Jerripothula and Vrishabhdhwaj Maharshi and Kashish Katara},
keywords = {Underwater enhancement, Multi-object tracking, Siamese network, Similarity},
abstract = {Tracking marine life plays a crucial role in understanding migration patterns, movements, and population growth of underwater species. Deep learning-based fish-tracking networks have been actively researched and developed, yielding promising results. In this work, we propose an end-to-end deep learning framework for tracking fish in unconstrained marine environments. The core innovation of our approach is a Siamese-based architecture integrated with an image enhancement module, designed to measure appearance similarity effectively. The enhancement module consists of convolutional layers and a squeeze-and-excitation block, pre-trained on degraded and clean image pairs to address underwater distortions. This enhanced feature representation is leveraged within the Siamese framework to compute an appearance similarity score, which is further refined using prediction scores based on fish movement patterns. To ensure robust tracking, we combine the appearance similarity score, prediction score, and IoU-based similarity score to generate fish trajectories using the Hungarian algorithm. Our framework significantly reduces ID switches by 35.6% on the Fish4Knowledge dataset and 3.8% on the GMOT-40 fish category, all while maintaining high tracking accuracy. The source code of this work is available here: https://github.com/srimanta-mandal/Multi-Fish-Tracking-with-Underwater-Image-Enhancement.}
}
```



## 📁 Dataset Preparation

To increase the sample size of each class in the **F4K dataset**, various **data augmentation** techniques were applied. This helps improve the robustness and performance of the tracking model.

## 🏋️‍♂️ Training the Model

You can train the Siamese-based model by specifying:

- The path to the dataset.
- The path to save the trained model.

Update the paths accordingly in the script:  
`ie_based_siamese.py`

**Run the following command to start training:**

```bash
python /{yourPath}/Multi-Fish-Tracking-with-Underwater-Image-Enhancement/networks/siamese/ie_based_siamese.py 
```

## 🧪 Testing the Model

Before testing:

- Add the path to the **saved model** or you can download [models](https://drive.google.com/drive/folders/1wYP71-ACfMqqS4-H38q1alWqkkfG0mB5?usp=sharing) folder containing trained models.
- Specify the path to **save result text files** from the tracker.

**Run the tracker:**

```bash
python tracker.py
```
Evaluate result text files on the corresponding video data and find MOTA and HOTA metrics.
