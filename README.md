> ## ⚠️ This is an adapted fork, not the original MorphoGen
>
> **Upstream:** [Brainsmatics/MorphoGen](https://github.com/Brainsmatics/MorphoGen) — *MorphoGen:
> Efficient Unconditional Generation of Long-Range Projection Neuronal Morphology via a
> Global-to-Local Framework*, Tianfang Zhu, Hongyang Zhou, Anan Li, **ICCV 2025**
> ([paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_MorphoGen_Efficient_Unconditional_Generation_of_Long-Range_Projection_Neuronal_Morphology_via_ICCV_2025_paper.pdf)).
> All credit for the method belongs to the original authors; the README below is theirs.
>
> This fork adapts the released code to run as a **baseline on MICrONS mouse-visual-cortex
> dendrites** rather than the whole-brain long-range projection neurons it was designed for.
> It is **not** a reimplementation and makes no claim to improve the method.
>
> **What differs from upstream** (full rationale in each commit message and in `RUN.md` §6):
> - **Runnability fixes.** `modules/functional/src` is absent from the release, so `import
>   DDPM_train` failed on any machine; the two ops DiT-3D needs now have pure-PyTorch
>   equivalents. Several other defects prevented training or generation from completing at all.
> - **Cortical-scale recalibration.** Constants tuned for millimetre-scale projection neurons
>   (soma-detection radius, short-branch pruning threshold) are wrong by orders of magnitude on
>   ~250 µm dendrites and are re-derived on training data.
> - **A `MorphoGen+` arm**, opt-in via flags. Shipped defaults are preserved, so the faithful
>   published behaviour is what runs unless those flags are passed.
> - **Evaluation tooling** (`tools/`) for comparison against a separate tree-generation model.
>
> Where the code and the paper disagree, this fork follows **the paper** and says so in the
> commit message.
>
> **Licence:** upstream ships no `LICENSE` file, so no redistribution terms are stated. This fork
> is shared for research reproducibility. Anyone reusing it should contact the original authors
> regarding licensing, and should cite the ICCV 2025 paper rather than this repository.

---

# MorphoGen
Efficient Unconditional Generation of Long-Range Projection Neuronal Morphology via a Global-to-Local Framework. 
Code is almost one-click runnable.
Below we introduce the environment dependencies, file description, datasets used, and the code execution. Paper is [here](https://iccv.thecvf.com/virtual/2025/poster/49). 

## Dependencies
python==3.8.5, pytorch==1.8.2, torchvision==0.9.2, cudatoolkit==11.1

See `requirements.txt` for detailed environment specifications.

## File Description
- `sub_process.py`: Converts raw SWC files to standardized point cloud data.
- `distort.py`: Distorts true branches to learn the mapping back to original state.
- `DDPM_train.py`: Trains the denoising diffusion probabilistic model to predict global structures.
- `Auxiliary_train.py`: Trains the auxiliary CNN networks to optimize the local structures.
- `morphology_gen.py`: Generates new morphology point clouds and converts into SWC files.

## Dataset
Long-range neuronal data is sourced from [this study](https://www.nature.com/articles/s41593-022-01041-5).  

- **CT subtypes (45-52)**: 1,085 neurons (all subtypes)  
- **PT subtypes (57-64)**: 1,005 neurons  
- **IT subtypes (34-44)**: 985 neurons  

## Code Execution
train the [DDPM](https://github.com/DiT-3D/DiT-3D)：
```
python DDPM_train.py --dataroot ${dataroot} --model_dir${model_dir} --device ${device}
```
train the Auxiliary CNN：
```
python Auxiliary_train.py
```
generate new neuron morphology:
```
python morphology_gen.py --dataroot ${dataroot} --model${model} --device ${device} --generate_dir ${generate_dir}
```

## Citation
If you find this repository useful, please cite our paper:  
```
InProceedings{Zhu_2025_ICCV,  
    author    = {Zhu, Tianfang and Zhou, Hongyang and Li, Anan},  
    title     = {MorphoGen: Efficient Unconditional Generation of Long-Range Projection Neuronal Morphology via a Global-to-Local Framework},  
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
    month     = {October},  
    year      = {2025},  
    pages     = {13021-13031}  
}
``` 
  
## Acknowledgement
Thanks for the wonderful work [DiT-3D](https://github.com/DiT-3D/DiT-3D).
