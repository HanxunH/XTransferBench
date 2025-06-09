# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP

<div align="center">
  <img src="assets/logo.jpg" alt="XTransfer Logo" style="width: 256px; height: auto;" />
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.05528" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/models?other=arxiv:2505.05528" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-blue.svg" alt="HuggingFace Community"></a>
  <a href="https://github.com/HanxunH/XTransferBench/blob/main/LICENSE" target="_blank"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
  <a><img alt="Made with Python" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

Code for ICML2025 Paper ["X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP"](https://arxiv.org/pdf/2505.05528)

---

## X-TransferBench
**X-TransferBench** is an open-source benchmark that offers a comprehensive collection of Universal Adversarial Perturbations (UAPs) capable of achieving super adversarial transferability. These UAPs can simultaneously transfer **across data distributions, domains, model architectures**, and **downstream tasks**. In essence, they are perturbations that can convert virtually *any input sample* into an adversarial example—effective against *any model* and *any task*.

---
## Installation

```shell
git clone https://github.com/hanxunh/XTransferBench.git
cd XTransferBench
pip3 install .
```

---
## Usage
```python
import XTransferBench
import XTransferBench.zoo

# List threat models
print(XTransferBench.zoo.list_threat_model())

# List UAPs under L_inf threat model
print(XTransferBench.zoo.list_attacker('linf_non_targeted'))

# Load X-Transfer with the Large search space (N=64) non-targeted
attacker = XTransferBench.zoo.load_attacker('linf_non_targeted', 'xtransfer_large_linf_eps12_non_targeted')

# Perturbe images to adversarial example
images = # Tensor [b, 3, h, w]
adv_images = attacker(images) 
```

---
## Demo

We provide a web demo using X-TransferBench that allows you to transform any image into an adversarial example using our curated collection of UAPs and TUAPs. You can access the demo at the link below. Once generated, the adversarial example can be tested on any model and task of your choice.

- [Huggingface Spaces](https://huggingface.co/spaces/hanxunh/XTransferBench-UAP-Linf)

---
## UAPs/TUAPs Collections
- **L_inf Non-Targeted:** Refer to [collections/l_inf_non_targeted.md](collections/l_inf_non_targeted.md) for configuration details.
- **L_inf Targeted:** Refer to [collections/l_inf_targeted.md](collections/l_inf_targeted.md) for configuration details.
- **L_2 Non-Targeted:** Refer to [collections/l_2_non_targeted.md](collections/l_2_non_targeted.md) for configuration details.
- **L_2 Targeted:** Refer to [collections/l_2_targeted.md](collections/l_2_targeted.md) for configuration details.


---
## Reproduce results from the paper


The repository includes sample code and all necessary files to reproduce the results reported in the paper.

For evaluation instructions, please refer to [evaluations/README.md](evaluations/README.md).

For generating UAPs/TUAPs, see [xtransfer/README.md](xtransfer/README.md).


---

## Security and Ethical Use Statement

**The perturbations provided in this project are intended solely for research purposes.** They are shared with the academic and research community to advance understanding of super transferable attacks and defenses.

Any other use of the data, model weights, or methods derived from this project, including but not limited to unauthorized access, modification, or malicious deployment, is strictly prohibited and not endorsed by this project. The authors and contributors of this project are not responsible for any misuse or unethical applications of the provided resources. Users are expected to adhere to ethical standards and ensure that their use of this research aligns with applicable laws and guidelines.

---

## Citation

```bibtex
@inproceedings{
  huang2025xtransfer,
  title={X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP},
  author={Hanxun Huang and Sarah Erfani and Yige Li and Xingjun Ma and James Bailey},
  booktitle={ICML},
  year={2025},
}
```

--- 
