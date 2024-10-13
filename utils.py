import torch


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