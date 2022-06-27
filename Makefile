all:
	srun --container-name=python3.8 -p cpu --pty python main.py