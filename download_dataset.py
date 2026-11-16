# %%
from datasets import load_dataset

# %%
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
print(dataset)
print(dataset["train"][0])

# %%
df = dataset["train"].to_pandas()
df.info()
print(df.head())