# Exact global explanations of piecewise-linear deep feedforward neural networks via rule extraction
This repository contains supplementary material for the paper:


Carles-Bou, J. L., & Carmona, E. J. (2026). **Exact global explanations of piecewise-linear deep feedforward neural networks via rule extraction**. Under revision.

## Paper Abstract
Deep feedforward neural networks (DFNNs) have achieved remarkable performance across a wide range of applications, yet their opaque decision-making processes hinder adoption in high-stakes domains where transparency, accountability, and regulatory compliance are essential. Although global explainability in neural networks has long been pursued through rule extraction techniques and, more recently, through aggregation of local explanations, existing approaches typically rely on approximation, sampling, or heuristic procedures, limiting faithful characterization of overall model behavior. This paper introduces G-FACE, an exact framework for global explanation in DFNNs with piecewise-linear (PWL) activation functions. Exploiting the continuous piecewise-affine structure of these networks, G-FACE transforms a trained model into an explicit rule-based knowledge representation composed of IF--THEN rules, each associated with an exact local affine mapping over a convex polyhedral activation region. A compact closed-form matrix formulation is derived to separate model parameters from input-dependent components, enabling exact and hyperparameter-free feature-level attribution. Unlike prior exact PWL approaches largely restricted to ReLU networks, the proposed framework extends to other PWL activations, including Leaky ReLU, hard sigmoid, and hard tanh. Beyond interpretability, the extracted rule-based representation supports advanced tasks such as local single- and multi-objective optimization and local adversarial search. Experimental use cases demonstrate the practical utility of G-FACE as a transparent and operational knowledge representation of trained neural networks.


## XX Repository description
The repository contains the source files developed to implement the new local post-hoc explainer for Feed-forward Neural 
Networks (FNNs) named "**Feature Attribution Computed Exactly (FACE)**" and the code of different examples showing the 
use of the explainer over regression and classification tabular problems and in classifation with images.

It has been implemented in Python and uses the Keras and Tensorflow framework. We are working in a new version compatible
with PyTorch, too.

The [*/source/mlpxai/explainers/face*](src/mlpxai/explainers/face) folder keeps the source files of the new explainer.

The [*/source/examples*](src/examples) folder contains Python programs that show how to use FACE.

And, in the [*/source/examples/notebooks*](src/examples/notebooks) folder, you can find the same Python programs but in Jupyter Notebook format:
- Tabular classification example ([notebook](https://github.com/CarlesBou/mlpxai/blob/main/src/examples/notebooks/TabularClassification.ipynb))
- Tabular regression example ([notebook](https://github.com/CarlesBou/mlpxai/blob/main/src/examples/notebooks/TabularRegression.ipynb))
- Image classification example ([notebook](https://github.com/CarlesBou/mlpxai/blob/main/src/examples/notebooks/MNISTClassification.ipynb))


## XX Instalation

Today, the better option to get the FACE code and start using it is to clone this Github. Simply clone the project with:

```sh
git clone https://github.com/CarlesBou/mlpxai.git
```



## Licenses
This work is licensed under Creative Commons Zero v1.0 Universal (or any later version) license.