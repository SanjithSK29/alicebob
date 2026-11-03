from datasets import load_dataset
from huggingface_hub import hf_hub_download

ds = load_dataset("SALT-NLP/agent-collusion", "episodes", split="test")
df = ds.to_pandas()
pilot = df[df.condition_id.str.contains("controlled_peer__gemini-3-1-flash-lite__summary--accept--full-feedback")]
pilot.to_parquet("pilot_episodes.parquet")
print(pilot.shape, pilot.both_accept.mean(), pilot.alice_verdict.value_counts())

for tf in pilot.trace_file.unique()[:5]:
    hf_hub_download("SALT-NLP/agent-collusion", tf, repo_type="dataset", local_dir="traces")
