import os
import numpy as np
from grakel import GraphletSampling

def calc_GS(adjacency_matrices, max_vertices, dir_name):
    os.makedirs(dir_name, exist_ok=True)

    data = [each_data[np.newaxis, np.newaxis, :] for each_data in adjacency_matrices]

    for i, i_data in enumerate(data):
        text = ''
        if i_data.shape[2] > max_vertices:
            continue

        # GS can be made in the outside for loop.
        gs_kernel = GraphletSampling(random_state=0, sampling={'epsilon': 0.05, 'delta': 0.05}, k=5)
        gs_kernel.fit(i_data)

        for j, j_data in enumerate(data):
            if j_data.shape[2] > max_vertices:
                continue
            if i > j:
                continue
            print(f'i: {i}, j: {j}')

            kernel_value = gs_kernel.transform(j_data)[0, 0]

            print(f'{kernel_value}')
            text += f'{i}-{j}\n{kernel_value}\n'
        with open(f'{dir_name}/{i}.pack', 'w') as f:
            f.write(text)
