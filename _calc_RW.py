import os
import numpy as np
from grakel import RandomWalk

def calc_RW(adjacency_matrices, max_vertices, dir_name):
    os.makedirs(dir_name, exist_ok=True)

    data = [each_data[np.newaxis, np.newaxis, :] for each_data in adjacency_matrices]

    for i, i_data in enumerate(data):
        text = ''
        if i_data.shape[2] > max_vertices:
            continue
        for j, j_data in enumerate(data):
            if j_data.shape[2] > max_vertices:
                continue
            if i > j:
                continue
            print(f'i: {i}, j: {j}')

            rw_kernel = RandomWalk()
            rw_kernel.fit(i_data)
            kernel_value = rw_kernel.transform(j_data)[0, 0]

            print(f'{kernel_value}')
            text += f'{i}-{j}\n{kernel_value}\n'
        with open(f'{dir_name}/{i}.pack', 'w') as f:
            f.write(text)
