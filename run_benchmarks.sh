python sharding.py --shards 1 --total-docs 4000000
python sharding.py --shards 2 --total-docs 4000000
python sharding.py --shards 4 --total-docs 4000000
python sharding.py --shards 8 --total-docs 4000000
python sharding.py --shards 1 --total-docs 500000
python sharding.py --shards 2 --total-docs 1000000
python sharding.py --shards 4 --total-docs 2000000
python sharding.py --shards 8 --total-docs 4000000

# python sharding.py --shards 1 --total-docs 4000000 --disable-np-multiprocessing
# python sharding.py --shards 2 --total-docs 4000000 --disable-np-multiprocessing
# python sharding.py --shards 4 --total-docs 4000000 --disable-np-multiprocessing
# python sharding.py --shards 8 --total-docs 4000000 --disable-np-multiprocessing
# python sharding.py --shards 1 --total-docs 500000 --disable-np-multiprocessing
# python sharding.py --shards 2 --total-docs 1000000 --disable-np-multiprocessing
# python sharding.py --shards 4 --total-docs 2000000 --disable-np-multiprocessing
# python sharding.py --shards 8 --total-docs 4000000 --disable-np-multiprocessing

# python sharding.py --shards 1 --computation-type matrix_mul_equal --total-docs 3000 --disable-np-multiprocessing
# python sharding.py --shards 4 --computation-type matrix_mul_equal --total-docs 3000 --disable-np-multiprocessing
# python sharding.py --shards 8 --computation-type matrix_mul_equal --total-docs 3000 --disable-np-multiprocessing
# python sharding.py --shards 12 --computation-type matrix_mul_equal --total-docs 3000 --disable-np-multiprocessing
# python sharding.py --shards 16 --computation-type matrix_mul_equal --total-docs 3000 --disable-np-multiprocessing
# python sharding.py --shards 1 --computation-type matrix_mul_unequal --total-docs 2000000 --disable-np-multiprocessing
# python sharding.py --shards 4 --computation-type matrix_mul_unequal --total-docs 2000000 --disable-np-multiprocessing
# python sharding.py --shards 8 --computation-type matrix_mul_unequal --total-docs 2000000 --disable-np-multiprocessing
# python sharding.py --shards 12 --computation-type matrix_mul_unequal --total-docs 2000000 --disable-np-multiprocessing
# python sharding.py --shards 16 --computation-type matrix_mul_unequal --total-docs 2000000 --disable-np-multiprocessing

# python sharding.py --shards 1 --total-docs 4000000 --computation-type annlite
# python sharding.py --shards 2 --total-docs 4000000 --computation-type annlite
# python sharding.py --shards 4 --total-docs 4000000 --computation-type annlite
# python sharding.py --shards 8 --total-docs 4000000 --computation-type annlite
# python sharding.py --shards 1 --total-docs 500000 --computation-type annlite
# python sharding.py --shards 2 --total-docs 1000000 --computation-type annlite
# python sharding.py --shards 4 --total-docs 2000000 --computation-type annlite
# python sharding.py --shards 8 --total-docs 4000000 --computation-type annlite
