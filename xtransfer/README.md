## Generate Attacker Configuration Files (*.yaml)
1.	Open `exp_config_template.py` and replace all `TODO: placeholders` with the appropriate file paths for your local environment.
2.	Edit `search/search_space.py` and set the `CACHE_DIR` variable to the directory where you want to store the OpenCLIP model weights.
3.	Modify `generate_exp_configs.py` to toggle between evaluation and attacker configurations by commenting or uncommenting lines marked with `#`.
4.	Run `generate_exp_configs.py` to generate the final configuration files.

- An example configuration is available in [l_inf_pgd_12](configs/untargeted/in1k/large/mab_ucb_rho2/Ensemble16/l_inf_pgd_12.yaml).

## Generate UAP/TUAP with XTransfer Attack
Our default implementation uses PyTorch Distributed Data Parallel (DDP) in a SLURM environment. The hyperparameter `k`—which determines the number of surrogate models selected per iteration—is controlled via the `WORLD_SIZE` environment variable. By default, the implementation loads exactly one model per GPU (i.e., per rank).

```shell
export MASTER_PORT={Port}
export "WORLD_SIZE="$SLURM_NTASKS
echo "WORLD_SIZE="$WORLD_SIZE
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
echo "MASTER_ADDR="$MASTER_ADDR
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

srun python3 generate_universal_perturbation.py --ddp --dist_eval                                   \
                                                --exp_name Which Attacker to use e.g. l_inf_pgd_12  \
                                                --exp_path PATH/TO/EXP_FILE_FOLDER                  \ 
                                                --exp_config PATH/TO/CONFIG/FOLDER         
```
