# Consolidated imports
import os
import re
import pandas as pd
from collections import defaultdict
from gseapy.biomart import Biomart
import gseapy as gp
from myvariant import MyVariantInfo
import argparse
import sys

# Retrieve the list of available Enrichr libraries (gp.get_library_name returns a list)
# libraries = gp.get_library_name()
# print("Available Enrichr libraries:")
# for lib in libraries:
#     print(lib)
# sys.exit(0)

# Parse command-line options
parser = argparse.ArgumentParser(description="Run ORA or GSEA on ranked gene list")
parser.add_argument(
    '--method',
    choices=['ora', 'gsea'],
    default='ora',
    help='Enrichment analysis method: "ora" for Over-Representation Analysis, "gsea" for Gene Set Enrichment Analysis'
)
parser.add_argument(
    '--cutoff', type=float, default=0.05,
    help='FDR cutoff for ORA (ignored for GSEA)'
)
parser.add_argument(
    '--out_dir', type=str,
    default="/Users/nickq/Documents/Pioneer Academics/Research_Project/src/classification_pipeline/post_analysis/results",
    help='Base output directory'
)
args = parser.parse_args()
# ensure base output dir exists
os.makedirs(args.out_dir, exist_ok=True)


def debug_mapping_stats(transcriptomics_raw, transcriptomics, ensg_map, transcriptomics_genes, rs_ids, jhu_ids,
                        rs_mapped, jhu_mapped):
    print("=== DEBUG MAPPING STATS ===")
    print(f"Total ranked features: {len(ranked_features)}")
    print(f"Transcriptomics raw: {len(transcriptomics_raw)}, stripped: {len(transcriptomics)}")
    print(f"ENSG map entries: {len(ensg_map)}")
    missing = [raw for raw, ensg in zip(transcriptomics_raw, transcriptomics) if ensg not in ensg_map]
    print(f"Unmapped ENSG IDs: {len(missing)} (first 5: {missing[:5]})")
    valid_genes = [g for g in transcriptomics_genes if isinstance(g, str) and not pd.isna(g)]
    print(f"Mapped transcriptomic genes: {len(valid_genes)} (first 5: {valid_genes[:5]})")
    print(f"Total SNPs: {len(rs_ids) + len(jhu_ids)} (rs: {len(rs_ids)}, jhu: {len(jhu_ids)})")
    print(f"Mapped rsIDs: {len(rs_mapped)}, JHU IDs: {len(jhu_mapped)}")
    print("==========================")


# Load ranked features
ranked_df = pd.read_csv(
    "/Users/nickq/Documents/Pioneer Academics/Research_Project/data/results/single_gcn_model_tweaking/thres_-4/80f/thres_tuning_sparse_50n/metrics_and_features/GCNN_fused_thr-4_sparse_final_feature_ranking.csv")  # Adjust the path

# Extract feature names
ranked_features = ranked_df['Feature'].tolist()

# --- Detect MeanImportance and StdImportance columns robustly ---
mean_col = None
std_col = None
for col in ranked_df.columns:
    low = col.lower().replace(" ", "").replace("_", "")
    if ("meanimportance" in low or ("importance" in low and "std" not in low)):
        mean_col = col
    if ("stdimportance" in low or ("std" in low and "importance" in low)):
        std_col = col


if mean_col:
    feature_mean_map = dict(zip(ranked_df['Feature'], ranked_df[mean_col]))
    print(f"Using '{mean_col}' for mean importance values.")
else:
    feature_mean_map = {}
    print("Warning: MeanImportance column not detected.")

if std_col:
    feature_std_map = dict(zip(ranked_df['Feature'], ranked_df[std_col]))
    print(f"Using '{std_col}' for std importance values.")
else:
    feature_std_map = {}
    print("Warning: StdImportance column not detected.")

# --- Hold out any "age" feature ---
age_features = [f for f in ranked_features if f.lower().startswith("age")]
total_features = len(ranked_features)
age_rev_score = None
age_mean_imp = None
age_std_imp = None
if age_features:
    feat = age_features[0]
    idx = ranked_features.index(feat)
    age_rev_score = total_features - idx
    age_mean_imp = feature_mean_map.get(feat)
    age_std_imp = feature_std_map.get(feat) if std_col else None
    # remove age from further feature processing
    ranked_features = [f for f in ranked_features if not f.lower().startswith("age")]

# Extract Ensembl IDs (remove version suffix if present)
transcriptomics_raw = [f for f in ranked_features if f.startswith("ENSG")]
transcriptomics = [ensg.split('.')[0] for ensg in transcriptomics_raw]

print("Mapping Ensembl gene IDs to gene symbols...")
ensg_symbols = Biomart().query(dataset='hsapiens_gene_ensembl',
                               attributes=['ensembl_gene_id', 'external_gene_name'])
ensg_map = dict(zip(ensg_symbols['ensembl_gene_id'], ensg_symbols['external_gene_name']))
transcriptomics_genes = [ensg_map[ensg] for ensg in transcriptomics if ensg in ensg_map]
print(f"Mapped {len(transcriptomics_genes)} transcriptomics features to gene symbols.")

# Filter out any unmapped or NaN gene symbols
valid_transcriptomics_genes = [g for g in transcriptomics_genes if isinstance(g, str) and not pd.isna(g)]

snps = [f for f in ranked_features if not f.startswith("ENSG") and not f.lower().startswith("age")]

print(f"Total features: {len(ranked_features)}")
print(f"Transcriptomics (ENSG): {len(transcriptomics)}")
print(f"SNPs: {len(snps)}")

# SNP mapping preparation

# Clean SNPs (remove any trailing spaces etc.)
snps = [s.strip() for s in snps]

# Categorize SNPs
rs_ids = [s for s in snps if s.startswith("rs")]
jhu_ids = [s for s in snps if s.startswith("JHU_")]

print(f"Total SNPs: {len(snps)}")
print(f" - RS IDs: {len(rs_ids)}")
print(f" - JHU IDs: {len(jhu_ids)}")

# SNP-to-gene mapping and BED export

mv = MyVariantInfo()

# Map rsIDs to genes, handling both dict and list structures
print("Querying rsIDs for gene mapping...")
rs_results = mv.getvariants(rs_ids, fields='dbsnp.gene')
rs_mapped = {}
for r in rs_results:
    if 'dbsnp' not in r or 'gene' not in r['dbsnp']:
        continue
    gene_info = r['dbsnp']['gene']
    symbol = None
    if isinstance(gene_info, dict):
        # single entry
        symbol = gene_info.get('symbol')
    elif isinstance(gene_info, list) and gene_info:
        # list of entries—take first dict or string
        first = gene_info[0]
        if isinstance(first, dict):
            symbol = first.get('symbol')
        elif isinstance(first, str):
            symbol = first
    if symbol:
        rs_mapped[r['query']] = symbol

print(f"Successfully mapped {len(rs_mapped)} rsIDs to gene symbols")

# --- Map JHU SNPs to genes using MyVariantInfo ---
print("Mapping JHU SNPs to gene symbols...")
hgvs_ids, jhu_query_map = [], {}
for jhu_id in jhu_ids:
    match = re.match(r"JHU_([A-Za-z0-9]+)\.(\d+)", jhu_id)
    if match:
        chrom, pos = match.groups()
        hgvs = f"{chrom}:g.{pos}"
        hgvs_ids.append(hgvs)
        jhu_query_map[hgvs] = jhu_id
jhu_results = mv.getvariants(hgvs_ids, fields='ensembl.gene')
jhu_mapped = {}
for r in jhu_results:
    query = r['query']
    gene = None
    if 'ensembl' in r and 'gene' in r['ensembl']:
        entry = r['ensembl']['gene']
        if isinstance(entry, list):
            gene = entry[0].get('symbol') if isinstance(entry[0], dict) else entry[0]
        elif isinstance(entry, dict):
            gene = entry.get('symbol')
    orig_jhu = jhu_query_map.get(query)
    if gene and orig_jhu:
        jhu_mapped[orig_jhu] = gene
print(f"Successfully mapped {len(jhu_mapped)} JHU IDs to gene symbols")
# Extend rs_mapped with JHU mappings for enrichment
rs_mapped.update(jhu_mapped)

# --- Nearest-gene fallback for any unmapped JHU SNPs ---
print("Applying nearest-gene fallback for unmapped JHU SNPs...")
# Retrieve gene TSS coordinates via BioMart
gene_attrs = ['chromosome_name', 'start_position', 'external_gene_name']
gene_info_df = Biomart().query(
    dataset='hsapiens_gene_ensembl',
    attributes=gene_attrs
)
# Build gene coordinate list
gene_list = [
    {'gene': row['external_gene_name'],
     'chrom': str(row['chromosome_name']),
     'pos': int(row['start_position'])}
    for _, row in gene_info_df.iterrows()
    if str(row['chromosome_name']).isdigit()
]
# Identify unmapped JHU IDs
unmapped_jhu = [jid for jid in jhu_ids if jid not in jhu_mapped]
mapped_count = 0
for jid in unmapped_jhu:
    match = re.match(r"JHU_([0-9]+)\.(\d+)", jid)
    if not match:
        continue
    chrom, pos = match.groups()
    pos = int(pos)
    same_chr = [g for g in gene_list if g['chrom'] == chrom]
    if not same_chr:
        continue
    # Find nearest gene by transcription start site distance
    nearest = min(same_chr, key=lambda g: abs(g['pos'] - pos))
    jhu_mapped[jid] = nearest['gene']
    mapped_count += 1
print(f"Nearest-gene fallback mapped {mapped_count} JHU IDs to gene symbols.")
# Merge fallback mappings into overall SNP-to-gene map
rs_mapped.update(jhu_mapped)

debug_mapping_stats(
    transcriptomics_raw,
    transcriptomics,
    ensg_map,
    transcriptomics_genes,
    rs_ids,
    jhu_ids,
    rs_mapped,
    jhu_mapped
)

print("Computing gene-level reverse ranking...")

# --- Build gene-level reverse rank and importance summaries ---
gene_data = defaultdict(lambda: {
    "reverse_scores": [],
    "importances": [],
    "std_importances": []
})
total_features = len(ranked_features)

for i, feat in enumerate(ranked_features):
    # Determine gene mapping
    gene = None
    if feat in ensg_map:
        gene = ensg_map[feat]
    elif feat in rs_mapped:
        gene = rs_mapped[feat]
    elif feat in jhu_mapped:
        gene = jhu_mapped[feat]

    if gene:
        # Reverse rank score: higher for more important features
        rev_score = total_features - i
        gene_data[gene]["reverse_scores"].append(rev_score)
        # Original importance, if available
        if mean_col:
            imp = feature_mean_map.get(feat)
            if imp is not None:
                gene_data[gene]["importances"].append(imp)
        # Original std, if available (including zero)
        if std_col and feat in feature_std_map:
            std_imp = feature_std_map[feat]
            # Check for numeric (not None or NaN)
            if pd.notnull(std_imp):
                gene_data[gene]["std_importances"].append(std_imp)

# Summarize per-gene: max reverse score and mean original importance and std
ranked_genes = []
for gene, vals in gene_data.items():
    max_rev = max(vals["reverse_scores"])
    mean_imp = (sum(vals["importances"]) / len(vals["importances"])
                if vals["importances"] else None)
    mean_std = (sum(vals["std_importances"]) / len(vals["std_importances"])
                if vals["std_importances"] else None)
    ranked_genes.append((gene, max_rev, mean_imp, mean_std))

 # Sort genes by descending reverse rank
ranked_genes.sort(key=lambda x: x[1], reverse=True)

# --- Append age back into the ranked list if present ---
if age_rev_score is not None:
    ranked_genes.append(("Age", age_rev_score, age_mean_imp, age_std_imp))
    # re-sort so Age is placed correctly by reverse score
    ranked_genes.sort(key=lambda x: x[1], reverse=True)

# --- Filter out genes with NaN or empty names ---
cleaned_ranked_genes = [
    (g, rev, mi, ms) for (g, rev, mi, ms) in ranked_genes
    if isinstance(g, str) and g.strip() and not pd.isna(g)
]
ranked_genes = cleaned_ranked_genes


# --- Write full metrics for genes with non-null StdImportance ---
filtered = [(g, rev, mi, ms) for (g, rev, mi, ms) in ranked_genes if ms is not None]
std_output_path = os.path.join(args.out_dir, "ranked_genes_with_std_importance.rnk")
with open(std_output_path, "w") as f:
    f.write("Gene\tReverseRankScore\tMeanImportance\tMeanStdImportance\n")
    for gene, rev_score, mean_imp, mean_std in filtered:
        imp_str = f"{mean_imp:.6f}" if mean_imp is not None else ""
        std_str = f"{mean_std:.6f}" if mean_std is not None else ""
        f.write(f"{gene}\t{rev_score}\t{imp_str}\t{std_str}\n")
print(f"Saved ranked genes with std_importance ➜ {std_output_path}")

output_path = os.path.join(args.out_dir, "ranked_genes_with_importance.rnk")
with open(output_path, "w") as f:
    f.write("Gene\tReverseRankScore\tMeanImportance\tMeanStdImportance\n")
    for gene, rev_score, mean_imp, mean_std in ranked_genes:
        imp_str = f"{mean_imp:.6f}" if mean_imp is not None else ""
        std_str = f"{mean_std:.6f}" if mean_std is not None else ""
        f.write(f"{gene}\t{rev_score}\t{imp_str}\t{std_str}\n")
print(f"Saved ranked gene list with importance and std to {output_path}")

# Prepare gene list for enrichment analyses, including only valid gene symbols
cleaned_genes = ranked_genes
gene_list = [
    str(gene) for gene, rev_score, mean_imp, mean_std in cleaned_genes
    if isinstance(gene, str) and gene
]
print(f"Prepared gene list for enrichment; sample entries: {gene_list[:5]}")

# Select libraries for enrichment
# libraries = [
#     "MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_Pathways_2024", "WikiPathway_2024_Human", "DisGeNET",
#     "OMIM_Disease", "GWAS_Catalog_2023", "GTEx_Tissues_V8_2023", "Allen_Brain_Atlas_10x_scRNA_2021", "SynGO_2024"
# ]

# if args.method == 'ora':
#     print("Running ORA with Enrichr...")
#     out_dir = os.path.join(args.out_dir, 'ORA_results')
#     os.makedirs(out_dir, exist_ok=True)
#     enr_res = gp.enrichr(
#         gene_list=gene_list,
#         gene_sets=libraries,
#         organism='Human',
#         outdir=out_dir,
#         cutoff=args.cutoff
#     )
#     ora_summary_csv = os.path.join(out_dir, "ORA_enrichr_summary.csv")
#     enr_res.results.to_csv(ora_summary_csv, index=False)
#     print(f"ORA completed. Summary saved to: {ora_summary_csv}")
#
# elif args.method == 'gsea':
#     print("Running GSEA prerank...")
#     out_dir = os.path.join(args.out_dir, 'GSEA_results')
#     os.makedirs(out_dir, exist_ok=True)
#     # Use the .rnk file created above
#     rnk_file = output_path  # path to ranked_genes_with_importance.rnk
#     gs_res = gp.prerank(
#         rnk=rnk_file,
#         gene_sets=libraries,
#         organism='Human',
#         outdir=out_dir,
#         threads=4,
#         min_size=5,
#         max_size=500
#     )
#     gsea_summary_csv = os.path.join(out_dir, "GSEA_prerank_results.csv")
#     gs_res.res2d.to_csv(gsea_summary_csv)
#     print(f"GSEA prerank completed. Results saved to: {gsea_summary_csv}")
