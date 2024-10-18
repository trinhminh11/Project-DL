import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def set_random_seed(seed: int) -> None:
	"""
	Sets the seeds at a certain value.
	:param seed: the value to be set
	"""
	print("Setting seeds ...... \n")
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
     
	

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def plotting(filename = None, **kwargs):
    for name, value in kwargs.items():
        plt.plot(value, label=name)
        plt.title(name)

    plt.legend()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
        
		