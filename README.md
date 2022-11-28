# Learning Green's Functions of Linear Reaction-Diffusion Equations with Application to Fast Numerical Solver
Code repository for the paper:  
**Learning Green's Functions of Linear Reaction-Diffusion Equations with Application to Fast Numerical Solver**  
[Yuankai Teng](https://slooowtyk.github.io), [Xiaoping Zhang](http://xpzhang.me/), [Zhu Wang](https://people.math.sc.edu/wangzhu), [Lili Ju](https://people.math.sc.edu/ju)

Proceedings of Third Mathematical and Scientific Machine Learning Conference (MSML'2022), 2022 <br>
[[paper](https://msml22.github.io/msml22papers/MSML22_GFNet.pdf)]


## Training Usage
To train the GF-Nets for a specific problem on given domain
```shell
python ./trainer.py 
--pde_case Poisson 
--domain_type square 
--blocks_num 4 4
--z_coarse_num 545  
--x_coarse_num 2105 
--x_middle_num 8265 
--x_refine_num 32753 
--scale 5 10 
--epochs_Adam 20000 
--epochs_LBFGS 10000 
--tol 1e-4 
--cuda_index 0
```
To finetune your model with smaller sigma
(when you have already trained your model with greater value of simga)
```shell
python ./trainer.py 
--pde_case Poisson 
--domain_type square 
--blocks_num 4 4
--z_coarse_num 545  
--x_coarse_num 2105 
--x_middle_num 8265 
--x_refine_num 32753 
--scale 5 10 
--sigma 0.015
--scale_resume 5 10
--sigma_resume 0.02
--resume True
--epochs_Adam 20000 
--epochs_LBFGS 10000 
--tol 1e-4 
--cuda_index 0
```

## Testing  Usage
To evaluate numerical error and draw your results:
```shell
python ./GreenFormula.py
--pde_case Poisson 
--solution_case 6
--domain_type washer 
--blocks_num 4 4
--z_coarse_num 545  
--x_coarse_num 2105 
--x_middle_num 8265 
--x_refine_num 32753 
--scale 5 10
```


## Citation
If you  find the idea or code of this paper useful for your research, please consider citing us:

```bibtex
@inproceedings{teng2022learning,
  title={Learning Green’s Functions of Linear Reaction-Diffusion Equations with Application to Fast Numerical Solver},
  author={Teng, Yuankai and Zhang, Xiaoping and Wang, Zhu and Ju, Lili},
  booktitle={Mathematical and Scientific Machine Learning},
  pages={1--16},
  year={2022},
  organization={PMLR}
}
```
