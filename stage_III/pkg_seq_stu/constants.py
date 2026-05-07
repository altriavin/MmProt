esm_root_path = "facebook/esm2_t33_650M_UR50D"
saport_root_path = "westlake-repl/SaProt_650M_AF2"

pretrain_root_path = "/root/autodl-tmp/MmProt/stage_II/toy/"
data_root_path = "/root/huxiaowen/mut_llm/protein_benchmark/metl_indepent_wt"

# DMS dataset configuration
dms_csv_path = "/root/autodl-tmp/MmProt/stage_III/toy/toy.csv"

# Few-shot configurations
FEW_SHOT_SIZES = [8, 16, 32, 64, 128, 256]

AA_ORDER = "LAGVSERTIDPKQNFYMHWC"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
