import time
import random

from jina import Executor, requests, Flow, DocumentArray
import numpy as np
from annlite import AnnLite

TOP_K = 96
EMBEDDING_DIM = 512
QUERIES_PER_REQUEST = 1
BENCHMARK_QUERIES = QUERIES_PER_REQUEST * 5
WARMUP_QUERIES = QUERIES_PER_REQUEST * 5


def get_random_embeddings(n_docs: int):
    docs = DocumentArray.empty(n_docs)
    docs.embeddings = np.random.random((n_docs, EMBEDDING_DIM)).astype(np.float32)
    return docs


def get_annlite_index(num_docs: int):
    da = DocumentArray.empty(num_docs)
    da.embeddings = np.random.random((num_docs, EMBEDDING_DIM)).astype(np.float32)

    random.randint(0, 1000000)
    ann = AnnLite(
        EMBEDDING_DIM,
        metric='cosine',
        data_path=f"/tmp/annlite_data{random.randint(0,1000000)}",
    )
    ann.index(da)
    return ann


def benchmark(
    total_docs: int,
    shards: int,
    computation_type: str,
    disable_np_multiprocessing: bool,
):
    if 'matrix_mul' not in computation_type and (
        total_docs % shards != 0 or TOP_K % shards != 0
    ):
        raise ValueError('total_docs and TOP_K must be divisible by shards')

    class Indexer(Executor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if computation_type == 'docarray_find':
                num_docs = int(total_docs // shards)
                self.da = DocumentArray.empty(num_docs)
                self.da.embeddings = np.random.random((num_docs, EMBEDDING_DIM)).astype(
                    np.float32
                )
            elif computation_type == 'annlite':
                self.ann = get_annlite_index(int(total_docs // shards))

        @requests
        def foo(self, docs, tracing_context, parameters, **kwargs):
            if computation_type == 'matrix_mul_equal':
                mat_A = np.random.random((total_docs, total_docs))
                mat_B = np.random.random((total_docs, total_docs))
            elif computation_type == 'matrix_mul_unequal':
                mat_A = np.random.random((total_docs, 100))
                mat_B = np.random.random((100, 100))

            with self.tracer.start_span('search', context=tracing_context) as span:
                span.set_attribute('shards', shards)
                span.set_attribute('total_docs', total_docs)
                span.set_attribute('computation_type', computation_type)
                span.set_attribute(
                    'disable_np_multiprocessing', disable_np_multiprocessing
                )
                span.set_attribute('request_type', parameters['request_type'])
                limit = int(TOP_K // shards)
                if computation_type == 'docarray_find':
                    return self.da.find(docs, limit=limit, metric='cosine')[0]
                elif computation_type == 'annlite':
                    return self.ann.search(docs, limit=limit)
                elif 'matrix_mul' in computation_type:
                    np.matmul(mat_A, mat_B)

    f = Flow(
        protocol='grpc',
        traces_exporter_host='0.0.0.0',
        traces_exporter_port=4317,
        tracing=True,
    ).add(
        uses=Indexer,
        traces_exporter_host='0.0.0.0',
        traces_exporter_port=4317,
        tracing=True,
        shards=shards,
        polling='all',
    )

    warmup_queries = get_random_embeddings(WARMUP_QUERIES)
    benchmark_queries = get_random_embeddings(BENCHMARK_QUERIES)

    with f:
        f.post(
            '/',
            warmup_queries,
            request_size=QUERIES_PER_REQUEST,
            parameters={'request_type': 'warmup'},
        )

        st = time.perf_counter()
        f.post(
            '/',
            benchmark_queries,
            request_size=QUERIES_PER_REQUEST,
            parameters={'request_type': 'benchmark'},
        )
        elapse = time.perf_counter() - st

        print('time for query', elapse)
        qps = BENCHMARK_QUERIES / elapse

        with open('benchmark.csv', 'a') as fp:
            fp.write(
                f'{disable_np_multiprocessing};{computation_type};{total_docs};{shards};{elapse:.2f};{qps:.0f}\n'
            )
