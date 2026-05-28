import pdb
import os
import csv
import clearml

class CSVLogger():
    def __init__(self, output_dir, ex_id):
        self.log_dir = os.path.join(output_dir, f"{ex_id}_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self._files = set()
    
    def report_scalar(self, title, series, value, iteration):
        '''
        Mimics clearml report_scalar() function to log values to CSV file
        '''
        if 'train' in series:
            filepath = os.path.join(self.log_dir, f"{title}_train.csv")
        else:
            filepath = os.path.join(self.log_dir, f"{title}_val.csv")

        write_header = filepath not in self._files

        with open(filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            if 'MEAN' in title:
                if write_header:
                    writer.writerow(["Series", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series, iteration, value])
            else:
                if write_header:
                    writer.writerow(["Fold", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series.split(' ')[-1], iteration, value])

def get_logger(
    logger_type: str,
    project_name: str,
    task_name: str,
    task_type: str = 'training',
    tags: list = None,
    config_dict: dict = None,
    output_dir: str = None,
    add_unique_id: bool = False,
):
    """
    Initialize a logger for experiment tracking.

    Args:
        logger_type: 'clearml' or 'csv'
        project_name: Name of the project (for ClearML)
        task_name: Name of the task/experiment
        task_type: Type of task - 'training', 'testing', 'inference', etc.
        tags: List of tags for the task (ClearML only)
        config_dict: Configuration dictionary to log (ClearML only)
        output_dir: Output directory (CSV only)
        add_unique_id: If True, append timestamp to task_name for uniqueness

    Returns:
        Logger instance (ClearML Logger or CSVLogger)
    """
    # Add unique identifier if requested
    if add_unique_id:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = f"{task_name}_{timestamp}"

    if logger_type == 'clearml':
        # Map task_type string to ClearML TaskTypes enum
        task_type_map = {
            'training': clearml.TaskTypes.training,
            'testing': clearml.TaskTypes.testing,
            'inference': clearml.TaskTypes.inference,
            'data_processing': clearml.TaskTypes.data_processing,
            'application': clearml.TaskTypes.application,
            'monitor': clearml.TaskTypes.monitor,
            'controller': clearml.TaskTypes.controller,
            'optimizer': clearml.TaskTypes.optimizer,
            'service': clearml.TaskTypes.service,
            'qc': clearml.TaskTypes.qc,
            'custom': clearml.TaskTypes.custom,
        }

        clearml_task_type = task_type_map.get(task_type, clearml.TaskTypes.training)

        task = clearml.Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=clearml_task_type,
            tags=tags or [],
            auto_connect_frameworks=False,
        )

        # Connect config if provided
        if config_dict is not None:
            task.connect(config_dict, name="config")

        logger = clearml.Logger.current_logger()
        return logger, task

    elif logger_type == 'csv':
        if output_dir is None:
            raise ValueError("output_dir must be provided for CSV logger")
        logger = CSVLogger(output_dir, task_name)
        return logger, None

    else:
        raise ValueError(f"Unknown logger type: {logger_type}. Use 'clearml' or 'csv'.")
