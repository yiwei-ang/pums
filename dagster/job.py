from dagster import job, op, In, Out, Nothing
import os
import sys
import subprocess


@op(
    config_schema={
        "person_table": str,
        "housing_table": str,
        "output_file": str
    },
    out=Out(Nothing)
)
def prepare_data_op(context):
    cfg = context.op_config

    try:
        result = subprocess.run(
            [
                sys.executable, "engine/data_preparation.py",
                "--person_table", cfg["person_table"],
                "--housing_table", cfg["housing_table"],
                "--output_file", cfg["output_file"]
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        context.log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        context.log.error(e.stderr)
        raise


@op(ins={"_start": In(Nothing)},
    config_schema={
        "input_file": str,
        "output_file": str
    },
    out=Out(Nothing)
    )
def run_inference_op(context):
    cfg = context.op_config
    try:
        result = subprocess.run(
            [
                sys.executable, "engine/data_inference.py",
                "--input_file", cfg["input_file"],
                "--output_file", cfg["output_file"]
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        context.log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        context.log.error(e.stderr)
        raise


@op(
    config_schema={
        "file": str,
        "bucket": str,
        "key": str
    },
    ins={"_start": In(Nothing)},
    out=Out(Nothing)
)
def upload_s3_op(context):
    cfg = context.op_config
    # script_path = os.path.abspath("engine/utils/s3_uploader.py")
    try:
        result = subprocess.run(
            [
                sys.executable, "engine/utils/s3_uploader.py",
                "--file", cfg["file"],
                "--bucket", cfg["bucket"],
                "--key", cfg["key"]
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        context.log.info(result.stdout)
    except subprocess.CalledProcessError as e:
        context.log.error(e.stderr)
        raise


@job
def snap_inference_job():
    step1 = prepare_data_op()
    step2 = run_inference_op(step1)
    upload_s3_op(step2)
