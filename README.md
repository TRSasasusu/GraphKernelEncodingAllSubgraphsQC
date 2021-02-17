# Numerical experiments for Graph kernel encoding all subgraphs by superposition of quantum computing

## Requirements

* python >=3.6
* pip3
* CUDA
* NVIDIA GPU with >=32510MiB memory

## Install

```bash
git clone https://github.com/TRSasasusu/GraphKernelEncodingAllSubgraphsQC
cd GraphKernelEncodingAllSubgraphsQC
pip3 install -r requirements.txt
```

If needed, use virtualenv.

## Usage

```bash
python3 calc_kernel.py <dataset> QK_BH
python3 calc_kernel.py <dataset> RW
python3 calc_kernel.py <dataset> GS
python3 calc_kernel.py <dataset> SP
python3 compare_kernel.py <dataset>
```

For <dataset>, select from MUTAG, AIDS, ER\_MD, PTC\_FM, BZR\_MD, IMDB-BINARY, Fingerprint and IMDB-MULTI.
