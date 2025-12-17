rsync --include="*/" --include="*.npz" --exclude="*" -amzP vulcan:/home/stang5/scratch/plant-rl/results .
rsync -azP vulcan:/home/stang5/scratch/plant-rl/results/offline/S8/P2/InAC/2357 results/offline/S8/P2/InAC/
rsync -azP vulcan:/home/stang5/scratch/plant-rl/results/offline/S8/P2/InAC_Calibration/664 results/offline/S8/P2/InAC_Calibration/