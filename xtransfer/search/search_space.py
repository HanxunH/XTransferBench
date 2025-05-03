import collections
from pprint import pprint
CACHE_DIR = "TODO:PATH_TO_CACHE_DIR"  # Set this to your cache directory

search_space_base = [
    # 4x ResNet
    {
      "model_name": "RN101",
      "pretrained": "yfcc15m",
    },
    {
      "model_name": "RN50",
      "pretrained": "yfcc15m",
    },
    {
      "model_name": "RN50",
      "pretrained": "cc12m",
    },
    {
      "model_name": "RN101-quickgelu",
      "pretrained": "yfcc15m",
    },
    # 4x ConvNext
    {
      "model_name": "convnext_base",
      "pretrained": "laion400m_s13b_b51k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k_augreg",
    },
    {
      "model_name": "convnext_large_d",
      "pretrained": "laion2b_s26b_b102k_augreg",
    },
    # 4x ViT-B
    {
      "model_name": "ViT-B-16",
      "pretrained": "dfn2b",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion2b_s34b_b79k",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    # 4x ViT-L
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_s13b_b90k",
    },
    {
      "model_name": "EVA02-L-14",
      "pretrained": "merged2b_s4b_b131k",
    },
    {
      "model_name": "ViT-L-14-CLIPA",
      "pretrained": "datacomp1b",
    },
]

for item in search_space_base:
    item['cache_dir'] = CACHE_DIR + '/{}_{}'.format(item['model_name'], item['pretrained'])
a = [item["model_name"]+"_"+item["pretrained"] for item in search_space_base]
print([item for item, count in collections.Counter(a).items() if count > 1])
print("search_space_base", len(set([item["model_name"]+"_"+item["pretrained"] for item in search_space_base])), len(search_space_base))

a = ['({}, {})'.format(item["model_name"], item["pretrained"]) for item in search_space_base]
pprint(a)

search_space_mid = [
    # ResNet x3
    {
      "model_name": "RN50",
      "pretrained": "cc12m",
    },
    {
      "model_name": "RN50",
      "pretrained": "yfcc15m",
    },
    {
      "model_name": "RN101",
      "pretrained": "yfcc15m",
    },
    

    # ConvNext
    {
      "model_name": "convnext_base",
      "pretrained": "laion400m_s13b_b51k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k_augreg",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion_aesthetic_s13b_b82k",
    },
    {
      "model_name": "convnext_large_d",
      "pretrained": "laion2b_s26b_b102k_augreg",
    },
    {
      "model_name": "convnext_xxlarge",
      "pretrained": "laion2b_s34b_b82k_augreg",
    },

    # ViT-B-32
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion2b_e16",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion2b_s34b_b79k",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "datacomp_xl_s13b_b90k",
    },

    # ViT-B-16
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion2b_s34b_b88k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "datacomp_l_s1b_b8k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "dfn2b",
    },
    {
      "model_name": "EVA02-B-16",
      "pretrained": "merged2b_s8b_b131k",
    },

    # ViT-L-14
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion2b_s32b_b82k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_clip_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_laion_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "dfn2b",
    },
    {
      "model_name": "EVA02-L-14",
      "pretrained": "merged2b_s4b_b131k",
    },
    {
      "model_name": "ViT-SO400M-14-SigLIP",
      "pretrained": "webli",
    },
    {
      "model_name": "ViT-L-14-CLIPA",
      "pretrained": "datacomp1b",
    },

]

for item in search_space_mid:
    item['cache_dir'] = CACHE_DIR + '/{}_{}'.format(item['model_name'], item['pretrained'])

a = [item["model_name"]+"_"+item["pretrained"] for item in search_space_mid]
print([item for item, count in collections.Counter(a).items() if count > 1])
print("search_space_mid", len(set([item["model_name"]+"_"+item["pretrained"] for item in search_space_mid])), len(search_space_mid))
a = ['({}, {})'.format(item["model_name"], item["pretrained"]) for item in search_space_mid]
pprint(a)

search_space_large = [
    # ResNet x3
    {
      "model_name": "RN50-quickgelu",
      "pretrained": "cc12m",
    },
    {
      "model_name": "RN50-quickgelu",
      "pretrained": "yfcc15m",
    },
    {
      "model_name": "RN101-quickgelu",
      "pretrained": "yfcc15m",
    },
    

    # 5x ConvNext
    {
      "model_name": "convnext_base",
      "pretrained": "laion400m_s13b_b51k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion2b_s13b_b82k_augreg",
    },
    {
      "model_name": "convnext_base_w",
      "pretrained": "laion_aesthetic_s13b_b82k",
    },
    {
      "model_name": "convnext_large_d",
      "pretrained": "laion2b_s26b_b102k_augreg",
    },
    {
      "model_name": "convnext_xxlarge",
      "pretrained": "laion2b_s34b_b82k_augreg",
    },
    {
      "model_name": "convnext_xxlarge",
      "pretrained": "laion2b_s34b_b82k_augreg_rewind",
    },
    {
      "model_name": "convnext_xxlarge",
      "pretrained": "laion2b_s34b_b82k_augreg_soup",
    },

    # ViT-B-32
    {
      "model_name": "ViT-B-32-quickgelu",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-B-32-quickgelu",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion2b_e16",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "laion2b_s34b_b79k",
    },
    {
      "model_name": "ViT-B-32",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-B-32-quickgelu",
      "pretrained": "metaclip_400m",
    },
    {
      "model_name": "ViT-B-32-quickgelu",
      "pretrained": "metaclip_fullcc",
    },

    # ViT-B-16
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "laion2b_s34b_b88k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "datacomp_l_s1b_b8k",
    },
    {
      "model_name": "ViT-B-16",
      "pretrained": "commonpool_l_s1b_b8k",
    },
    {
      "model_name": "ViT-B-16-quickgelu",
      "pretrained": "dfn2b",
    },
    {
      "model_name": "ViT-B-16-quickgelu",
      "pretrained": "metaclip_400m",
    },
    {
      "model_name": "ViT-B-16-quickgelu",
      "pretrained": "metaclip_fullcc",
    },
    {
      "model_name": "EVA02-B-16",
      "pretrained": "merged2b_s8b_b131k",
    },

    # ViT-L-14
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion400m_e31",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion400m_e32",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "laion2b_s32b_b82k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "datacomp_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_clip_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_laion_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14",
      "pretrained": "commonpool_xl_s13b_b90k",
    },
    {
      "model_name": "ViT-L-14-quickgelu",
      "pretrained": "metaclip_400m",
    },
    {
      "model_name": "ViT-L-14-quickgelu",
      "pretrained": "metaclip_fullcc",
    },
    {
      "model_name": "ViT-L-14-quickgelu",
      "pretrained": "dfn2b",
    },
    {
      "model_name": "EVA02-L-14",
      "pretrained": "merged2b_s4b_b131k",
    },
    {
      "model_name": "ViT-SO400M-14-SigLIP",
      "pretrained": "webli",
    },
    {
      "model_name": "ViT-L-14-CLIPA",
      "pretrained": "datacomp1b",
    },

    # ViT-H-14
    {
      "model_name": "ViT-H-14",
      "pretrained": "laion2b_s32b_b79k",
    },
    {
      "model_name": "ViT-H-14-quickgelu",
      "pretrained": "metaclip_fullcc",
    },
    {
      "model_name": "ViT-H-14-quickgelu",
      "pretrained": "dfn5b",
    },

    # ViT-g-14
    {
      "model_name": "ViT-g-14",
      "pretrained": "laion2b_s12b_b42k",
    },
    {
      "model_name": "ViT-g-14",
      "pretrained": "laion2b_s34b_b88k",
    },
    {
      "model_name": "EVA02-E-14-plus",
      "pretrained": "laion2b_s9b_b144k",
    },
    {
      "model_name": "EVA01-g-14-plus",
      "pretrained": "merged2b_s11b_b114k",
    },
    {
      "model_name": "ViT-bigG-14",
      "pretrained": "laion2b_s39b_b160k",
    },
    {
      "model_name": "ViT-bigG-14-CLIPA",
      "pretrained": "datacomp1b",	
    },
    # Roberta
    {
      "model_name": "roberta-ViT-B-32",
      "pretrained": "laion2b_s12b_b32k",
    },
    {
      "model_name": "xlm-roberta-base-ViT-B-32",
      "pretrained": "laion5b_s13b_b90k",
    },
    {
      "model_name": "xlm-roberta-large-ViT-H-14",
      "pretrained": "frozen_laion5b_s13b_b90k",
    },

    # NLLB
    {
      "model_name": "nllb-clip-base",
      "pretrained": "v1",
    },
    {
      "model_name": "nllb-clip-large",
      "pretrained": "v1",
    },
    
    # ViTamin
    {
      "model_name": "ViTamin-B",
      "pretrained": "datacomp1b",
    },
    {
      "model_name": "ViTamin-B-LTT",
      "pretrained": "datacomp1b",
    },
    {
      "model_name": "ViTamin-L",
      "pretrained": "datacomp1b",
    },
    {
      "model_name": "ViTamin-L2",
      "pretrained": "datacomp1b",
    },
    # MobileCLIP
    {
      "model_name": "MobileCLIP-S1",
      "pretrained": "datacompdr",
    },
    {
      "model_name": "MobileCLIP-S2",
      "pretrained": "datacompdr",
    },
    {
      "model_name": "MobileCLIP-B",
      "pretrained": "datacompdr",
    },
    # COCA
    {
      "model_name": "coca_ViT-L-14",
      "pretrained": "laion2b_s13b_b90k",
    },
    {
      "model_name": "coca_ViT-B-32",
      "pretrained": "laion2b_s13b_b90k",
    },
]
for item in search_space_large:
    item['cache_dir'] = CACHE_DIR + '/{}_{}'.format(item['model_name'], item['pretrained'])

a = [item["model_name"]+"_"+item["pretrained"] for item in search_space_large]
print([item for item, count in collections.Counter(a).items() if count > 1])


search_space_rsicd = [
    # 4x ResNet
    {
      "model_name": "huggingface:flax-community/clip-rsicd",
    },
    {
      "model_name": "huggingface:flax-community/clip-rsicd-v2",
    },
    {
      "model_name": "huggingface:flax-community/clip-rsicd-v3",
    },
    {
      "model_name": "huggingface:flax-community/clip-rsicd-v4",
    },
]

