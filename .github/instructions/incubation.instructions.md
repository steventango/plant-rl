First figure out what the next experiment number should be.
For example, if the "experiments/online" folder has "E11", it should be "E12".
Then use the previous experiment like "experiments/online/E11/P0" as a reference to create the next experiment: "experiments/online/E12/P0".
Make sure to update the "experiments/online/E12/P0/README.md" file inside the new experiment folder to reflect the changes.
Also, update "docker-compose.yml" to use the new experiment configs.
