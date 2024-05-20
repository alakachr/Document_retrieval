# Document_retrieval

The goal of this project is to train an embedding model that will be used to measure similarity among documents. Given a dataset made of triplets of texts (query, positive, negative), we want queries embeddings to be closer to their corresponding positive example than the negative one. This task is called semantic textual similarity (STS) and is useful for document retrieval, where we want to retrieve or rank documents base on their similarity with a given context or query.

Dependancies can be found in the [pyproject.toml](pyproject.toml) file.
