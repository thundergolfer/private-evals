import datetime
import os
import shutil
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("pyproject.toml", remote_path="/pyproject.toml", copy=True)
    .add_local_file(
        "templates/report_template.html", remote_path="/root/templates/report_template.html", copy=True
    )
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands("uv sync")
    .pip_install("fastapi", "uvicorn")
    .add_local_python_source(
        "tasks", "evaluator", "llm", "container_controller", "llm_tests"
    )
)
volume = modal.Volume.from_name("private-llm-benchmarks", create_if_missing=True)
app = modal.App(
    name="private-llm-benchmarks",
    image=image,
    secrets=[modal.Secret.from_name("private-llm-benchmark")],
)


@app.function(
    volumes={"/data/": volume},
    schedule=modal.Cron("0 0 * * 1"),  # At midnight on Monday.
)
def cron():
    from tasks import run_benchmarks, PROVIDER_TO_DEFAULT_MODEL

    results_src = "results.json"
    report_src = "report.html"
    run_benchmarks(PROVIDER_TO_DEFAULT_MODEL, results_src, report_src)
    now = datetime.datetime.now()
    datetime_prefix = now.strftime("%Y%m%d_%H%M%S")
    results_dst = f"/data/{datetime_prefix}_results.json"
    report_dst = f"/data/{datetime_prefix}_report.html"
    shutil.copy(results_src, results_dst)
    shutil.copy(report_src, report_dst)
    print(f"Copied results to {results_dst}")
    print(f"Copied report to {report_dst}")


@app.function()
@modal.fastapi_endpoint(label="private-llm-benchmarks")
def serve_static(path: str):
    if not path:
        path = "index.html"

    # Reject requests which don't know the secret path.
    # TODO: could use Modal's proxy auth.
    data_dir_prefix = os.environ["DATA_DIR_PREFIX_SECRET"]
    if path.startswith(data_dir_prefix):
        return {"error": f"File {path} not found"}, 404

    try:
        with open(path, "r") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return {"error": f"File {path} not found"}, 404
