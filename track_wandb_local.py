import os
import argparse
import time

def get_latest_wandb_subfolder(wandb_path="./wandb"):
    """
    Gets the path to the latest subfolder within the specified wandb directory.

    Args:
        wandb_path (str, optional): Path to the wandb directory. Defaults to "./wandb".

    Returns:
        str or None: Path to the latest subfolder, or None if no subfolders are found or wandb_path doesn't exist.
    """
    if not os.path.exists(wandb_path) or not os.path.isdir(wandb_path):
        print(f"Error: Wandb directory '{wandb_path}' not found or is not a directory.")
        return None

    subfolders = [
        os.path.join(wandb_path, d)
        for d in os.listdir(wandb_path)
        if os.path.isdir(os.path.join(wandb_path, d)) and "offline" in d
    ]

    if not subfolders:
        print(f"Error: No subfolders found in '{wandb_path}'.")
        return None

    # Get creation time for each subfolder (or modification time if creation time is not reliable)
    subfolders_with_time = []
    for folder in subfolders:
        try:
            timestamp = os.path.getctime(folder)  # Creation time (may not be reliable on all systems)
        except OSError: # Fallback to modification time if creation time fails
            timestamp = os.path.getmtime(folder)
        subfolders_with_time.append((timestamp, folder))

    # Sort by time (newest first)
    subfolders_with_time.sort(key=lambda x: x[0], reverse=True)

    return subfolders_with_time[0][1]  # Return the path of the latest subfolder


def rename_latest_wandb_subfolder(new_name, wandb_path="./wandb"):
    """
    Gets the latest subfolder in wandb and renames it to the provided new_name.

    Args:
        new_name (str): The new name for the latest subfolder.
        wandb_path (str, optional): Path to the wandb directory. Defaults to "./wandb".
    """
    latest_subfolder_path = get_latest_wandb_subfolder(wandb_path)

    if latest_subfolder_path:
        parent_dir = os.path.dirname(latest_subfolder_path)
        new_folder_path = os.path.join(parent_dir, new_name)

        try:
            os.rename(latest_subfolder_path, new_folder_path)
            print(f"Successfully renamed latest wandb subfolder '{os.path.basename(latest_subfolder_path)}' to '{new_name}'.")
        except OSError as e:
            print(f"Error renaming folder: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename the latest wandb subfolder to a specified name.")
    parser.add_argument("--filename", help="The new name for the latest wandb subfolder.")
    parser.add_argument("--wandb_path", default="./wandb", help="Path to the wandb directory (default: ./wandb)")

    args = parser.parse_args()

    rename_latest_wandb_subfolder(args.filename, args.wandb_path)
