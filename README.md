# Inverse Stochastic User Equilibrium with LOGIT assignment (isuelogit)

Stochastic user equilibrium with logit assignment (`sue-logit`) outputs an assignment of travelers in a transportation network where no individual believes that he can unilaterally improve his utility by choosing an alternative path. A key limitation of `sue-logit` is that it requires knowing the parameters of the traveler's utility function beforehand. 

The `isuelogit` package addresses the aforementioned limitation by using the output of `sue-logit` to solve an inverse problem, namely, estimating the parameters of the travelers' utility function using traffic counts. It also provides point estimates and hypothesis tests for the parameter estimates. To understand the theory behind the algorithms and the use cases of this package, you can review the following documents: 

+ Preprint: https://arxiv.org/abs/2204.10964
+ Journal article: https://doi.org/10.1016/j.trb.2023.102853

Please cite this work as:

```
@article{GuardaQian2024statistical,
  title={Statistical inference of travelers’ route choice preferences with system-level data},
  author={Guarda, Pablo and Qian, Sean},
  journal={Transportation research part B: methodological},
  volume={179},
  pages={102853},
  year={2024}
}
```

## Development setup

1. Clone the repository
2. Download and install Anaconda: https://docs.anaconda.com/anaconda/install/index.html
2. Create conda environment: `conda create -n isuelogit`
3. Activate environment: `conda activate isuelogit`
4. Install dependencies: `conda env update -f isuelogit.yml`
5. Install the package: `pip install -e .`
6. Install the development dependencies: `pip install -r requirements/dev.txt`
7. Run the tests: `pytest`

## Jupyter notebooks
1. Install jupyter lab: `pip install jupyter notebook`
2. Install ipython kernel for jupyter: `pip install ipykernel`
3. Add your virtual environment to Jupyter:  `python -m ipykernel install --user --name=isuelogit`
4. Open any of the notebooks included in the folder `notebooks` 
5. Choose kernel `isuelogit` before running the notebook

## Examples

The folder ``notebooks`` contains Jupyter notebooks with code demonstrations that can be reproduced from your local environment. If you are using VS Code, please make sure to select the ``isuelogit`` environment as your kernel to run each notebook.

## Collaboration
For any questions or interest in collaborating on this project, please open an issue in this repository. This package was developed under the guidance of Prof. Sean Qian. 

## Other notes
The `isuelogit` codebase includes several utilities to work with system-level data and transportation networks and it is further extended in [pesuelogit](https://github.com/pabloguarda/pesuelogit) to jointly estimate the Origin-Destination matrix using system-level data collected in multiple hourly periods and days. 

## Funding 

This project was partially funded through National Science Foundation Grant CMMI-1751448: [Probabilistic Network Flow Theory](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1751448)

## 🌟 Loved the Project? Give Us a Star!
We are thrilled to see you here! If you find this codebase useful for your project or it has been a helpful resource, please consider giving it a star. 🌟 Your support means a lot and it also helps others discover this work.