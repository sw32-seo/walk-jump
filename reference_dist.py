from walkjump.metrics import LargeMoleculeDescriptors

from tqdm import tqdm

import pandas as pd

from fast_edit_distance import edit_distance


def get_descriptors_as_dict(sequence: str) -> dict:
    return {k: v for k, v in LargeMoleculeDescriptors.from_sequence(sequence).asdict().items() if k in set(LargeMoleculeDescriptors.descriptor_names())}

def rename_df(df: pd.DataFrame, prefix: str):
    return df.rename({c: f"{prefix}_{c}" for c in df.columns}, inplace=True, axis=1)

df = pd.read_csv("data/pvrig.csv.gz")

tqdm.pandas(desc="heavy")
descriptor_df_heavy = pd.DataFrame.from_records(df.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
tqdm.pandas(desc="light")
descriptor_df_light = pd.DataFrame.from_records(df.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for light chains

rename_df(descriptor_df_heavy, "fv_heavy")
rename_df(descriptor_df_light, "fv_light")

ref_feats = pd.concat([descriptor_df_heavy, descriptor_df_light, df], axis=1)

sample_df = pd.read_csv("pvrig_denoise_samples.csv")

tqdm.pandas(desc="heavy")
samp_descriptor_df_heavy = pd.DataFrame.from_records(sample_df.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
tqdm.pandas(desc="light")
samp_descriptor_df_light = pd.DataFrame.from_records(sample_df.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for light chains

rename_df(samp_descriptor_df_heavy, "fv_heavy")
rename_df(samp_descriptor_df_light, "fv_light")

sample_df_with_descriptors = pd.concat([sample_df, samp_descriptor_df_heavy, samp_descriptor_df_light], axis=1)

from walkjump.metrics import get_batch_descriptors

description_heavy = get_batch_descriptors(sample_df_with_descriptors, ref_feats, "fv_heavy")
description_light = get_batch_descriptors(sample_df_with_descriptors, ref_feats, "fv_light")

print(description_heavy)
print(description_light)

heavy_uniqueness = 0
for i, row in sample_df.iterrows():
    # Check if the fv heavy chain is the same as in df
    uniqueness = len(df[df['fv_heavy_aho'] == row.fv_heavy_aho])
    if uniqueness > 0:
        heavy_uniqueness += 1

def min_edit_distance(sequence):
    # trastzumab = 'EVQLVES-GGGLVQPGGSLRLSCAASG-FNIKD-----TYIHWVRQAPGKGLEWVARIYPT---NGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDG-------------------FYAMDYWGQGTLVTVSS'
    edit_dist = df.apply(lambda x: edit_distance(x.fv_heavy_aho + x.fv_light_aho, sequence), axis=1).min()
    
    return edit_dist

tqdm.pandas(desc="min_edit_distance")
sample_df['min_edit_distance'] = sample_df.progress_apply(lambda row: min_edit_distance(row.fv_heavy_aho + row.fv_light_aho), axis=1)
print("Avg min edit distance =", sample_df['min_edit_distance'].mean())
print("Std min edit distance =", sample_df['min_edit_distance'].std())
print("Uniquness =", 1 - heavy_uniqueness/len(sample_df))
