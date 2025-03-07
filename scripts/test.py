import threading
import wandb
import time

# Function to run a wandb experiment
def run_experiment(project_name, run_name, num_steps=100):
    wandb.init(project=project_name, name=run_name)
    for step in range(num_steps):
        wandb.log({"step": step, "metric": step * 2})
        time.sleep(0.5)  # Simulate work
    wandb.finish()

# Thread targets with different project and run names
thread1 = threading.Thread(target=run_experiment, args=("project_a", "run_1"))
thread2 = threading.Thread(target=run_experiment, args=("project_a", "run_2"))

# Start threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

print("Both experiments finished.")