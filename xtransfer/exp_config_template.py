generation_template = {
  "epochs": 1,
  "log_frequency": 50,
  "sync_bn": False,
  "amp": False,
  "target_text": None,
  "attacker": {
    "name": "PGDLInfinity",
    "epsilon": 0.047058823529411764,
    "step_size": 0.00196078431372549,
    "target_text": "$target_text"
  },
  "dataset": {
    "name": "DatasetGenerator",
    "train_bs": 128,
    "eval_bs": 512,
    "n_workers": 8,
    "train_d_type": "ImageFolder",
    "test_d_type": "ImageFolder",
    "train_tf_op": "CLIPCC3M",
    "test_tf_op": "CLIPCC3M",
    "train_path": "TODO:/PATH/TO/ImageNet/ILSVRC/Data/CLS-LOC",
    "test_path": "/TODO:/PATH/TO/ImageNet/ILSVRC/Data/CLS-LOC",
    "collate_fn": {
      "name": "None"
    }
  }
}

zero_shot_eval_template = {
  "epochs": 1,
  "log_frequency": 200,
  "sync_bn": False,
  "zero_shot_templates": "CIFAR_TEMPLATES",
  "class_names": "CIFAR10_CLASSNAMES",
  "attacker_checkpoint_path": "experiments/template1/cc3m/ViT-L-14_laion400m_e32/generation/checkpoints",
  "normalize_image": True,
  "target_text": None,
  "amp": True,
  "model": {
    "name": "create_model_and_transforms",
    "model_name": "ViT-L-14",
    "pretrained": "laion400m_e32",
    "cache_dir": "TODO:/PATH/TO/cache/openclip/ViT-L-14_laion400m_e32"
  },
  "attacker": {
    "name": "TriggerGeneration",
    "alpha": 0.0001,
    "beta": 70,
    "target_text": "$target_text",
    "image_normalize": True
  },
  "dataset": {
    "name": "DatasetGenerator",
    "train_bs": 256,
    "eval_bs": 512,
    "n_workers": 8,
    "train_d_type": "ConceptualCaptionsDataset",
    "test_d_type": "CIFAR10",
    "train_tf_op": "CLIP",
    "test_tf_op": "CLIP",
    "train_path": "TODO:/PATH/TO/cc3m",
    "test_path": "TODO:/PATH/TO/datasets",
    "collate_fn": {
      "name": "None"
    }
  }
}
