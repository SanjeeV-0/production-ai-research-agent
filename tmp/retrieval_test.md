# Production Knowledge Systems

A production knowledge system is usually built from several components that work together. Documents enter the system through an ingestion process, are converted into smaller units, and are eventually made searchable. The quality of the final answers depends not only on the language model but also on how information is stored, divided, represented, retrieved, and ranked.

A useful system should preserve enough context around a piece of information so that the retrieved text remains meaningful when it is shown to a language model. At the same time, storing extremely large pieces of text can make retrieval less precise. This creates a practical tension between context and retrieval granularity.

## Retrieval-Augmented Generation

Retrieval-Augmented Generation, commonly called RAG, combines information retrieval with language generation. Instead of asking a language model to answer entirely from information encoded in its parameters, a RAG system first searches a knowledge base for relevant information and then provides the retrieved material to the model.

The basic flow is straightforward. A user asks a question, the system represents the question in a form that can be compared with stored information, relevant pieces of the knowledge base are retrieved, and the resulting context is passed to the language model. The model can then generate an answer grounded in the retrieved material.

RAG is particularly useful when the information changes over time or when an organization wants answers to be based on its own documents. Updating the underlying knowledge base can be easier than retraining a language model whenever a document changes.

There are several different retrieval strategies. Traditional keyword search can identify documents containing important terms. Dense retrieval instead represents text as vectors and searches for vectors that are close to the representation of the query.

Some systems combine both approaches. A keyword system can be useful when exact terminology matters, while dense retrieval can identify semantically related text even when the query and document use different words.

## Vector Representations

An embedding is a numerical representation of a piece of text. An embedding model converts text into a vector containing many numerical dimensions. Texts with related meanings can occupy nearby regions of the vector space.

For example, the sentences "A customer submitted a support request" and "A client opened a service ticket" use different words but describe similar situations. A semantic embedding model can represent their meanings in a way that allows a retrieval system to recognize their relationship.

Cosine similarity is one commonly used measure for comparing vectors. The retrieval system can calculate the similarity between a query vector and vectors stored for document chunks, then use the resulting distances or similarities to rank candidate chunks.

A vector database does not understand the text in the same way that a human does. It stores numerical representations and provides mathematical operations for comparing them. The meaning comes from the embedding model that produced the vectors.

In this project, PostgreSQL is used for relational storage and pgvector is used to store vector embeddings. A chunk can therefore retain its text and document relationships while also having a vector representation used during retrieval.

## Chunking and Document Structure

Chunking is one of the most important stages in a retrieval pipeline. A document that is several thousand words long is generally too large to treat as one retrieval unit. Dividing it into smaller pieces allows the retrieval system to return more focused context.

Simple systems sometimes split text after a fixed number of characters or tokens. This approach is easy to implement, but it can divide a concept in the middle of an explanation.

Semantic chunking takes a different approach. It attempts to preserve coherent groups of related content. Adjacent pieces of prose can be compared using embeddings, and sufficiently related pieces can remain together.

This does not mean that every semantically related statement in an entire document should become one chunk. Retrieval systems normally need reasonably local units. A statement in one chapter may be conceptually related to another statement many pages later, but combining them would create an unnecessarily large retrieval unit.

The system also needs to deal with headings, lists, tables, and other structural elements. A heading can provide important context even though it is not itself a normal paragraph. A table may need to remain intact because separating its rows from their headers can make the resulting text difficult to interpret.

A practical chunking pipeline therefore has to balance several goals:

- preserve semantic coherence
- retain document structure
- avoid excessively large chunks
- preserve page and section information
- produce units that are useful for retrieval

An additional consideration is that the same document may be processed more than once. If a document changes, a production system should be able to distinguish different versions rather than silently replacing historical data.

## An Unrelated Operational Note

The engineering team uses Docker for local development. Services can be started with Docker Compose so that developers do not have to install every infrastructure dependency directly on their machines.

Application logs are normally collected during development when debugging failures. A developer might inspect API logs, database connection messages, or model-loading messages when a service does not start correctly.

The team's internal meeting room is on the second floor near the main reception area. The cafeteria usually has several lunch options, and employees often check the daily menu before deciding whether to eat there or order food.

These operational details are not part of the retrieval architecture itself. They simply represent the type of unrelated information that can appear in a larger organizational knowledge base.

## Retrieval Pipeline

Once chunks have been created, the system generates an embedding for each final chunk and stores the vector alongside the chunk content. When a user submits a query, the same embedding model can convert the query into a vector.

The vector representation of the query is then compared with the stored chunk vectors. The closest candidates are selected according to the configured similarity measure.

Retrieval is normally a candidate-generation step rather than the final decision about what context should be shown to the model. A system may retrieve more candidates initially and then apply a reranker to determine which pieces are most useful.

For example, a query about vector databases could retrieve chunks discussing embeddings, cosine similarity, pgvector, and database storage. A reranker can then consider the actual query and candidate text together to produce a more refined ordering.

The number of candidates retrieved matters. Retrieving too few candidates can cause the correct information to be missed. Retrieving too many can introduce unnecessary context and increase the amount of information that the generation model must process.

## Reranking

Reranking is commonly applied after an initial retrieval stage. The first stage is optimized for efficiently finding potentially relevant candidates from a larger collection. The second stage can spend more computation examining a smaller candidate set.

A cross-encoder is one possible reranking model. Instead of independently representing a query and document and then comparing the two vectors, a cross-encoder receives both pieces of text together and produces a relevance score.

This can make reranking more expressive, although it is generally more expensive than a simple vector similarity operation. For that reason, vector search and reranking often serve different roles in the same pipeline.

A retrieval system should not assume that the highest vector similarity is always the best final result. Similarity is a useful signal, but relevance can depend on the exact question, document structure, freshness, and other metadata.

## Metadata and Traceability

The text of a chunk is only part of the information needed by a production knowledge system. The system should also retain information about where the chunk came from.

Useful metadata can include:

- document identifier
- document version
- section identifier
- section path
- page number
- chunk index
- source information

Section hierarchy can be useful because a chunk under "Retrieval" has different contextual meaning from an identically worded chunk under another part of a document.

Page information can also be important when a user needs to trace an answer back to the original source. In an enterprise setting, provenance can be as important as the generated answer itself.

The storage system therefore needs to preserve the relationship between documents, sections, pages, chunks, and embeddings rather than treating every vector as an isolated object.

## Evaluation

A retrieval system should be evaluated separately from the quality of the final language-model response. If the correct information is never retrieved, improving the generation prompt cannot fully solve the problem.

Retrieval evaluation can include measures such as recall and precision. Recall asks whether relevant information was retrieved, while precision considers how much of the retrieved material was actually relevant.

For a question-answering system, evaluation examples should contain a query and one or more pieces of information that are considered relevant. The retrieval system can then be tested to determine whether those pieces appear among the retrieved candidates.

Generation evaluation is a separate problem. An answer can be fluent but poorly grounded, or it can contain correct information while failing to explain the answer clearly.

A useful evaluation dataset should therefore cover straightforward questions, questions requiring information from different sections, ambiguous questions, and questions for which the knowledge base contains no suitable answer.

## Production Considerations

A prototype can often assume that documents are processed successfully. A production system has to account for failures, retries, duplicate requests, changing documents, and partial processing.

Document lifecycle state is useful for representing whether a document has been uploaded, is currently being processed, is ready for retrieval, or has failed processing.

Versioning becomes important when a document changes. A new version should be processed independently so that a currently searchable version is not destroyed before the replacement has completed successfully.

Idempotency is another important property. If the same content is submitted more than once for the same logical document, the system should be able to recognize that the content has already been processed rather than unnecessarily creating another identical version.

File storage is also separate from vector storage. The original uploaded file may need to be preserved for auditing, reprocessing, or future extraction. The database can retain metadata describing the file while the actual bytes remain in file storage.

Not every piece of information in a knowledge base is equally useful for every query. A good retrieval system therefore needs both strong candidate generation and sensible filtering. Document status and version information can prevent stale or failed material from appearing in normal retrieval results.

## A Small Note About Infrastructure

The research service is exposed through an API, and the application environment contains configuration for the database, embedding model, reranker, and language model provider. Development environments often use local services while production deployments may use managed infrastructure.

Python virtual environments make it possible to isolate project dependencies. Automated tests are useful for verifying ingestion, lifecycle transitions, retrieval behavior, and other important components before changes are deployed.

This information is mostly about software engineering practices rather than retrieval itself.

## Final Perspective

A production RAG system is therefore more than a language model connected to a vector database. It is a pipeline that preserves source documents, extracts structure, creates retrieval units, generates embeddings, stores provenance, retrieves candidate information, optionally reranks those candidates, and finally supplies selected context to a generation model.

The most important design decisions are often made before the language model receives a prompt. If the source document is poorly represented, incorrectly chunked, or incorrectly retrieved, the generation stage has limited information with which to produce a grounded answer.

At the same time, retrieval should remain connected to the underlying document model. Knowing which document, version, section, and page produced a chunk allows the system to reason about freshness and provenance rather than treating the vector collection as an anonymous set of numbers.