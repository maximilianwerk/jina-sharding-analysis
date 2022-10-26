import os
import click


@click.command()
@click.option('--disable-np-multiprocessing', is_flag=True)
@click.option('--total-docs', type=int, default=500000)
@click.option('--shards', type=int, default=1)
@click.option('--computation-type', default='docarray_find')
def main(disable_np_multiprocessing, total_docs, shards, computation_type):
    # disable these before importing numpy
    if disable_np_multiprocessing:
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    from benchmark import benchmark

    benchmark(total_docs, shards, computation_type, disable_np_multiprocessing)


if __name__ == '__main__':
    main()
