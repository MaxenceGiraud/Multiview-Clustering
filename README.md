# Multiview Clustering

Research project by [Maxence Giraud](https://github.com/MaxenceGiraud) on "highest order clustering" supervized by [Remy Boyer](https://pro.univ-lille.fr/remy-boyer/). 

We first implement algorithms of multi-view spectral clustering presented in [1].

## Installation 
### Requirements 
The requirements are precised in the requirements.txt file, to install them :
```bash
pip install -r requirements.txt
```

To install simply clone the project :
```bash
git clone https://github.com/MaxenceGiraud/Multiview-Clustering
cd Multiview-Clustering/
```
## Usage
```python3
import multiview_clustering as mc

# Compute your similarity matricies S = [S1,S2,...,Sk]
oi = mc.MC_FR_OI(n_clusters)
clusters= oi.fit_predict(S)
```

### Experiments
My experiments are compiled in the python file [experiments.py](./experiments.py), either run it using IPython or simply with python command : 

```bash
python3 experiments.py
```

## References

[1] Multi-View Partitioning via Tensor Methods. (2013) Xinhai Liu, Shuiwang Ji, Wolfgang Glänzel, and Bart De Moor, Fellow, IEEE       
