# Methods

One folder per method: `card.yaml`, `method.py`, `README.md` (its guide) and a test. A folder with only a `README.md` is a guide for code not written yet; the registry reads only folders with a `card.yaml`.

Methods come from `project_proposal.md` §4.1, plus the negative control.
Verdict (working note, 16 Nov 2026): A = use frozen as it is; B = frozen by default, fine-tuning possible; C = fitted on our data by design; D = not usable now.
Tier: v0 = built with the core; T1 = next; T2 = if time allows.
Status: `in progress` = being written with the core; `guide` = guide written, code to write; `to write` = no guide yet; `blocked` = verdict D, with the reason.

| Id | Name | Family | Verdict | Tier | Status |
|---|---|---|---|---|---|
| [`random_features`](random_features/) | Random features (negative control) | control | n/a | v0 | in progress |
| [`phikon_v2`](phikon_v2/) | Phikon-v2 | encoder, H&E | B | v0 | in progress |
| [`resnet50`](resnet50/) | ResNet50 (ImageNet) | encoder, H&E | A | v0 | in progress |
| `uni` | UNI | encoder, H&E | B | T1 | to write |
| [`uni2_h`](uni2_h/) | UNI2-h | encoder, H&E | B | T1 | guide |
| `conch` | CONCH | encoder, H&E | B | T2 | to write |
| `conch_v1_5` | CONCH v1.5 | encoder, H&E | A | T1 | to write |
| `virchow2` | Virchow2 | encoder, H&E | B | T1 | to write |
| `h_optimus_0` | H-optimus-0 | encoder, H&E | A | T1 | to write |
| `h_optimus_1` | H-optimus-1 | encoder, H&E | A | T2 | to write |
| `phikon` | Phikon | encoder, H&E | B | T2 | to write |
| `prov_gigapath` | Prov-GigaPath | encoder, H&E | B | T2 | to write |
| `ctranspath` | CTransPath | encoder, H&E | B | T2 | to write |
| `seal` | SEAL | encoder, H&E | A | T1 | to write |
| `h2o` | H2O | encoder, H&E | D? | T2 | blocked: weights and licence not confirmed, only the abstract could be read |
| `storm` | STORM | encoder, H&E | D | T2 | blocked: weights not released |
| `handcrafted_morphology` | Handcrafted morphology | encoder, H&E | C | T2 | to write |
| [`hvg_pca`](hvg_pca/) | HVG + PCA | encoder, RNA | C | v0 | in progress |
| [`scgpt`](scgpt/) | scGPT | encoder, RNA | B | T1 | guide |
| `geneformer` | Geneformer (V2) | encoder, RNA | B | T1 | to write |
| `uce` | UCE | encoder, RNA | A | T1 | to write |
| `cellplm` | CellPLM | encoder, RNA | B | T2 | to write |
| `nicheformer` | Nicheformer | encoder, RNA | B | T2 | to write |
| `scgpt_spatial` | scGPT-spatial | encoder, RNA | A | T1 | to write |
| `novae` | Novae | encoder, RNA | C on 55 µm Visium, B on Visium HD | T1 | to write |
| [`mean_intensity`](mean_intensity/) | Mean marker intensity | encoder, protein | C | v0 | in progress |
| `pca_protein` | Protein PCA | encoder, protein | C | T1 | to write |
| `astir` | astir | encoder, protein | C | T1 | to write |
| `maps` | MAPS | encoder, protein | C | T2 | to write |
| `kronos2` | KRONOS2 | encoder, protein | A | T1 | to write |
| `virtues` | VirTues-sp32 | encoder, protein | B | T1 | to write |
| [`eva`](eva/) | Eva (H&E and protein) | joint | B | T1 | guide |
| `rosie` | ROSIE | translator | B | T1 | to write |
| `gigatime` | GigaTIME | translator | B | T1 | to write |
| `miphei_vit` | MIPHEI-ViT | translator | B | T1 | to write |
| `histoplexer` | HistoPlexer | translator | D | T2 | blocked: the released model predicts a melanoma panel, 4 of our 11 markers |
| [`concat`](concat/) | Early fusion (concatenation) | fuser | A | v0 | in progress |
| `late_fusion` | Late fusion (prediction averaging) | fuser | C | T2 | to write |
| `cross_attention` | Cross-attention | fuser | C? | T2 | to write |
| [`spatialglue`](spatialglue/) | SpatialGlue | fuser | C on Simkin, Haist; D on Causer, CORAL | T1 | guide |
| `miso` | MISO | fuser | C on Simkin, Haist; D on Causer, CORAL | T1 | to write |
| `spamv` | SpaMV | fuser | C (same section) | T1 | to write |
| `totalvi` | totalVI | fuser | D | T2 | blocked: takes raw UMI counts only, from the same cells |
| `mofa_plus` | MOFA+ / MEFISTO | fuser | C | T1 | to write |
| `seurat_wnn` | Seurat WNN | fuser | A (graph output) | T1 | to write |
| `cosie` | COSIE | fuser | C | T1 | to write |
| `spamosaic` | SpaMosaic | fuser | C? | T2 | to write |
| `stabmap` | StabMap | fuser | C? | T2 | to write |
| `grover` | GROVER | fuser | C? | T2 | to write |
| `spemo` | spEMO | fuser | A (zero-shot; uses the OpenAI API) | T2 | to write |
| `cosmos` | COSMOS | fuser | C? | T2 | to write |
| `praga` | PRAGA | fuser | C | T2 | to write |
| `spamode` | SpaMode | fuser | C? | T2 | to write |
| `multivi` | MultiVI | fuser | D | T2 | blocked: counts only, built for RNA and ATAC |
| `hescape_contrastive` | HESCAPE-style contrastive alignment | fuser | C? | T2 | to write |
| `maxfuse` | MaxFuse | matcher | C? | T1 | to write |

PHARAOH (proposal §4.1, Tier 2) is not listed: it is a registration tool, so it belongs to data preparation (working note, 16 Nov 2026).

## Adding a method

1. Open a "Method" issue. If the method has no guide yet, copy [`_template/README.md`](_template/README.md) to `<method_id>/README.md` and fill it in.
2. Follow "Adding a method" in [`USBB_Codebase_README.md`](../../../USBB_Codebase_README.md): card, method, environment group, contract test.
3. Open the pull request, tick its "Adding a method" boxes, and update the method's row here.
