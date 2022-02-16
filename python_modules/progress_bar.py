"""

Progress bar

"""


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='x', print_end="\r"):
	if(total != 0):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration/float(total)))
		filled_length = int(length * iteration // total)
		bar = fill * filled_length + "-" * (length - filled_length)
		print(f"\r{prefix} |{bar}| {percent}% ({iteration}/{total}) {suffix}", end=print_end)
		if iteration == total:
			print()

