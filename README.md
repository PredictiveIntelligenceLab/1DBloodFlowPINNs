# Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks

Advances in computational science offer a principled pipeline for predictive modeling of cardiovascular flows and aspire to provide a valuable tool for monitoring, diagnostics and surgical planning. Such models can be nowadays deployed on large patient-specific topologies of systemic arterial networks and return detailed predictions on flow patterns, wall shear stresses, and pulse wave propagation. However, their success heavily relies on tedious pre-processing and calibration procedures that typically induce a significant computational cost, thus hampering their clinical applicability. In this work we put forth a machine learning framework that enables the seamless synthesis of non-invasive in-vivo measurement techniques and computational flow dynamics models derived from first physical principles. We illustrate this new paradigm by showing how one-dimensional models of pulsatile flow can be used to constrain the output of deep neural networks such that their predictions satisfy the conservation of mass and momentum principles. Once trained on noisy and scattered clinical data of flow and wall displacement, these networks can return physically consistent predictions for velocity, pressure and wall displacement pulse wave propagation, all without the need to employ conventional simulators. A simple post-processing of these outputs can also provide a relatively cheap and effective way for estimating Windkessel model parameters that are required for the calibration of traditional computational models. The effectiveness of the proposed techniques is demonstrated through a series of prototype benchmarks, as well as a realistic clinical case involving in-vivo measurements near the aorta/carotid bifurcation of a healthy human subject.

- Georgios Kissas, Yibo Yang, Eileen Hwuang, Walter R. Witschey, John A. Detre, Paris Perdikaris. "[Machine learning in cardiovascular flows modeling: Predicting pulse wave propagation from non-invasive clinical measurements using physics-informed deep learning.](https://www.sciencedirect.com/science/article/pii/S0045782519305055?dgcid=author)" (2019).


## Citation
```
@article{kissas2019machine,
title = "Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks",
journal = "Computer Methods in Applied Mechanics and Engineering",
volume = "358",
pages = "112623",
year = "2020",
issn = "0045-7825",
doi = "https://doi.org/10.1016/j.cma.2019.112623",
url = "http://www.sciencedirect.com/science/article/pii/S0045782519305055",
author = "Georgios Kissas and Yibo Yang and Eileen Hwuang and Walter R. Witschey and John A. Detre and Paris Perdikaris",
}
```
