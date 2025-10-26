"""Utility functions for syncing to GCS and other helpers."""

import os
from google.cloud import storage
import json


def sync_to_bucket(local_path, bucket_name='temporal-grounding-gpt2-82feb'):
    """
    Sync results to GCS bucket after each phase.

    Args:
        local_path: Local directory or file to sync
        bucket_name: GCS bucket name
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        if os.path.isfile(local_path):
            # Single file
            blob_name = os.path.basename(local_path)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            print(f"✓ Synced file: {blob_name}")
        else:
            # Directory
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    blob_path = os.path.relpath(local_file, os.path.dirname(local_path))
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_file)
                    print(f"✓ Synced: {blob_path}")

        print(f"✓ Sync complete to gs://{bucket_name}/")

    except Exception as e:
        print(f"⚠ Warning: Could not sync to bucket: {e}")
        print("Continuing with local files only...")


def save_checkpoint(data, filename, also_sync=True):
    """
    Save checkpoint and optionally sync to GCS.

    Args:
        data: Data to save (dict or object)
        filename: Path to save to
        also_sync: Whether to sync to GCS
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save based on extension
    if filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        # Assume pickle or numpy
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    print(f"✓ Saved: {filename}")

    if also_sync:
        sync_to_bucket(filename)
