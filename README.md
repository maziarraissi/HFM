[![DOI](https://zenodo.org/badge/152933799.svg)](https://zenodo.org/badge/latestdoi/152933799)

# [Hidden Fluid Mechanics](https://maziarraissi.github.io/HFM/)

We present [hidden fluid mechanics](https://science.sciencemag.org/content/367/6481/1026.abstract) (HFM), a physics informed deep learning framework capable of encoding an important class of physical laws governing fluid motions, namely the Navier-Stokes equations. In particular, we seek to leverage the underlying conservation laws (i.e., for mass, momentum, and energy) to infer hidden quantities of interest such as velocity and pressure fields merely from spatio-temporal visualizations of a passive scaler (e.g., dye or smoke), transported in arbitrarily complex domains (e.g., in human arteries or brain aneurysms). Our approach towards solving the aforementioned data assimilation problem is unique as we design an algorithm that is agnostic to the geometry or the initial and boundary conditions. This makes HFM highly flexible in choosing the spatio-temporal domain of interest for data acquisition as well as subsequent training and predictions. Consequently, the predictions made by HFM are among those cases where a pure machine learning strategy or a mere scientific computing approach simply cannot reproduce. The proposed algorithm achieves accurate predictions of the pressure and velocity fields in both two and three dimensional flows for several benchmark problems motivated by real-world applications. Our results demonstrate that this relatively simple methodology can be used in physical and biomedical problems to extract valuable quantitative information (e.g., lift and drag forces or wall shear stresses in arteries) for which direct measurements may not be possible.

For more information, please refer to the following: (https://maziarraissi.github.io/HFM/)

  - Raissi, Maziar, Alireza Yazdani, and George Em Karniadakis. "[Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations](https://science.sciencemag.org/content/367/6481/1026.abstract)." Science 367.6481 (2020): 1026-1030.

  - Raissi, Maziar, Alireza Yazdani, and George Em Karniadakis. "[Hidden Fluid Mechanics: A Navier-Stokes Informed Deep Learning Framework for Assimilating Flow Visualization Data](https://arxiv.org/abs/1808.04327)." arXiv preprint arXiv:1808.04327 (2018).

## Note

The required data (to be copied in the Data directory) and some Matlab scripts (to be copied in the Figures directory) for plotting purposes are provided in the following link:

   - [Data and Figures](https://bit.ly/2NRB65U)

In addition to the Data and Figures directories, the Results folder is currently empty and will be automatically populated after running the corresponding examples provided in the Source and Scripts directories.

## Citation

    @article{raissi2020hidden,
      title={Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations},
      author={Raissi, Maziar and Yazdani, Alireza and Karniadakis, George Em},
      journal={Science},
      volume={367},
      number={6481},
      pages={1026--1030},
      year={2020},
      publisher={American Association for the Advancement of Science}
    }

    @article{raissi2018hidden,
      title={Hidden Fluid Mechanics: A Navier-Stokes Informed Deep Learning Framework for Assimilating Flow Visualization Data},
      author={Raissi, Maziar and Yazdani, Alireza and Karniadakis, George Em},
      journal={arXiv preprint arXiv:1808.04327},
      year={2018}
    }
