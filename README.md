# Feature Enrichement
This repository includes feature enrichement approaches based on similar trajectories for pems station d1 traffic flow data.

To obtain similar trajectories, use the following files by changing window size (window_size), number of similar data (k) and metric:
    
- data_preparation_similar_trajectories.py 
- data_preparation_similar_trajectories_test.py

To obtain model result for xgboost, run xgboost_model.py with default features. If you want to add some features from similar trajectories, run file

- xgboost_model_with_targets_from_similar_trajectory.py to use targets from similar trajectories by changing number of targets.
- xgboost_model_with_avg_targets_from_similar_trajectory.py to use average of the targets from similar trajectories by changing number of targets.

