import matplotlib.pyplot as plt
import numpy as np
from tensorflow.io import write_file, gfile, read_file



def get_rate_schedule(rate_schedule, nlayers):
    schedule_name, schedule_args = rate_schedule
    if schedule_name == "shifted_exp":
        weights = np.exp(np.linspace(0, np.log(schedule_args["scale"]), sum(nlayers))) + schedule_args["shift"]
    elif schedule_name == "constant_per":
        assert len(nlayers) == len(schedule_args), f"the provided schedule has {len(schedule_args)} arguments but the model has only {len(nlayers)} different resolutions"
        weights = []
        for i in reversed(range(len(nlayers))):
            weights += [schedule_args[i]] * nlayers[i]
        weights = np.array(weights)

    weights = (weights / np.sum(weights) * 66.7).astype('float32')
    return weights

def get_resolutions(max_res, num_res):
    resos = []
    current_res = max_res
    for i in range(num_res):
        if current_res<4: current_res = 1
        resos.append(current_res)
        current_res //= 2
    return resos

def get_evenly_spaced_indices(N, K):
    #N=16, K=1: [8]
    #N=16, K=5: [3, 5, 8, 11, 13]
    #N=16, K=0: []   
    if K==0: return []

    insert_every_n = N / (K + 1)
    indices = [round(insert_every_n * i) for i in range(1, K+1)]
    return indices

#prints to the console and appends to the logfile at logfile_path
def print_and_log(*args, logfile_path):
    print(*args)
    for a in args:
        with gfile.GFile(logfile_path, mode='a') as f:
            f.write(str(a))

    with gfile.GFile(logfile_path, mode='a') as f:
        f.write('\n')

#postprocesses images for saving
def denormalize(ims, dtype='uint8'):
    return np.clip((ims+1)*127.5, 0, 255).astype(dtype)

#save images
def save_images(images, save_path, nrow=6, scale=5):
    if nrow is None:
        m = int(np.ceil(len(images)/10))
        n = 10
    else:
        m = nrow
        n = int(np.ceil(len(images)/nrow))

    plt.figure(figsize=(scale*n, scale*m))

    for i in range(len(images)):
        plt.subplot(m, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')

    plt.tight_layout()
    plt_savefig(save_path)
    plt.close('all')

#a helper function that acts like plt.savefig() except supports GCS file system
def plt_savefig(figure_path):
    if not (figure_path.startswith("gs://") or figure_path.startswith("gcs://")):
        plt.savefig(figure_path)
        return
    
    plt.savefig("./tmp_figure.png")

    write_file(
        figure_path, read_file("./tmp_figure.png")
    )
