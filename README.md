# Numerical experiments for Graph kernel encoding all subgraphs by superposition of quantum computing

## Requirements

* python >=3.6
* pip3
* CUDA
* NVIDIA GPU with >=32510MiB memory

## Installing

```bash
git clone https://github.com/TRSasasusu/GraphKernelEncodingAllSubgraphsQC
cd GraphKernelEncodingAllSubgraphsQC
pip3 install -r requirements.txt
cp grakel-datasets-base.py /usr/local/lib/python3.6/dist-packages/grakel/datasets/base.py
```

`grakel-datasets-base.py` fixes error when loading Fingerprint dataset by excluding graphs with \#edges less than 1.

If needed, use virtualenv.

## Usage

### Computing kernel values

```bash
python3 calc_kernel.py <dataset> <kernel>
```

Compute kernel values for each pair of graphs.  
For `<dataset>`, select from MUTAG, AIDS, ER\_MD, PTC\_FM, BZR\_MD, IMDB-BINARY, Fingerprint and IMDB-MULTI.  
For `<kernel>`, select from QK\_BH\_ve, QK\_BH\_ved, QK\_SH\_ve, QK\_SH\_ved, RW, GS and SP.

### Performance evaluation

```bash
python3 evaluate.py <dataset> <kernel>
```

Evaluate performance of the kernel.  
Make sure to run `calc_kernel.py` first.

The results are stored in `performance_results`.
