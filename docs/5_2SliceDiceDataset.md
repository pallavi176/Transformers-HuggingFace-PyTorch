# Time to slice and dice

## Slice and dice a dataset 

- Most of the time the data you work with, won't be perfectly prepared for training models.
- We will explore the various features that the datasets library provides to clean up your data.
- The Datasets library provides several methods to filter and transform a dataset.
    - Shuffle and split
    - Select and filter
    - Rename, remove and flatten
    - Map

### Shuffle and split

- You can easily shuffle the whole dataset with Dataset.shuffle()

``` py
from datasets import load_dataset

squad = load_dataset("squad", split="train")
squad[0]

squad_shuffled = squad.shuffle(seed=666)
squad_shuffled[0]
```

- It is genrally a good idea to place shuffling to your training set so that your model doesn't learn any artificial ordering the data.

- Another way to shuffle the data is to create random train and test splits.
- This can be useful if you have to create your own test splits from raw data.
- To do this, you just apply the train_test_split() and specify how large the test split should be.

``` py
dataset = squad.train_test_split(test_size=0.1)
dataset
```

### Select and filter

- You can return rows according to a list of indices using Dataset.select().
- This method expects a list or a generator of the dataset's indices and will then return a new dataset object containing just those rows.

``` py
indices = [0, 10, 20, 40, 80]
examples = squad.select(indices)
examples
```

- If you want to create a random sample of rows, you can do this by chaining the shuffle and select methods together.

``` py
sample = squad.shuffle().select(range(5))
sample
```

- The last way to pick out specific rows in a dataset is by applying the filter().
- This method checks whether each row fulfills some condition or not.

``` py
squad_filtered = squad.filter(lambda x : x["title"].startswith("L"))
squad_filtered[0]
```

### Rename, remove and flatten

- Use the rename_column() and remove_column() methods to transform your columns.
- rename_column() to change the name of the column.

``` py
squad.rename_column("context", "passages")
squad
```

- remove_column() to delete them

``` py
squad.remove_columns(["id", "title"])
squad
```

- Some datasets have nested columns and you can expand these by applying the flatten()

``` py
squad.flatten()
squad
```

### Map method

- the Dataset.map() method applies a custom processing function to each row in the dataset

``` py
def lowercase_title(example):
    return {"title": example["title"].lower()}

squad_lowercase = squad.map(lowercase_title)
# Peek at random sample
squad_lowercase.shuffle(seed=42)["title"][:5]
```

- The map() method can also be used to feed batches of rows to the processing function.
- This is especially useful for tokenization where the tokenizers are backed by the tokenizer library and they can use fast multithreading to process batches in parallel. 

``` py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_title(example):
    return tokenizer(example["title"])

squad.map(tokenize_title, batched=True, batch_size=500)
```

## From Datasets to DataFrames and back

``` py
from datasets import load_dataset

dataset = load_dataset("swiss_judgment_prediction", "all_languages", split="train")
dataset[0]
```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```

``` py

```