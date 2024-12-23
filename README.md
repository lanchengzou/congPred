# **Congestion Prediction**

This is the official open-source repository for **congestion prediction**, leveraging both **layout** and **netlist** information for accurate predictions.

------

## **Installation**

To set up the environment, use the following command:

```
conda env create --file environment.yml
```

------

## **Code Structure**

Here is a brief overview of the project's code structure:

- **`dataViG.py`**:
   Handles dataset processing, including data preparation and transformations.
- **`train_laynet.py`**:
   The main script for training the model.
- **`LayNet.py`**:
   Defines the customized model architecture, which includes swin transformer block and heterogeneous GNN layer. The features of layout and netlist will be fused and learnt by the model.
- **`utils/`**:
   Contains utility functions for:
  - Logging and recording results.
  - Evaluation metrics to assess model performance.