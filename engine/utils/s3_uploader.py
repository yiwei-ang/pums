import boto3
import os


def upload_to_s3(local_path: str, bucket: str, s3_key: str, aws_profile: str = None):
    """
    Upload a file to S3.

    Args:
        local_path (str): Path to local file (e.g., 'working_dir/scored_snap.parquet')
        bucket (str): S3 bucket name (e.g., 'my-data-bucket')
        s3_key (str): S3 key path (e.g., 'snap/predictions/scored_snap.parquet')
        aws_profile (str): Optional AWS CLI profile name
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client("s3")

    print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
    s3.upload_file(local_path, bucket, s3_key)
    print("Upload complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--profile", required=False, default=None)
    args = parser.parse_args()

    upload_to_s3(
        local_path=args.file,
        bucket=args.bucket,
        s3_key=args.key,
        aws_profile=args.profile
    )
# python engine/utils/s3_uploader.py --file working_dir/scored_snap.parquet --bucket food-stamp-prediction --key snap_outputs/ca/scored.parquet

