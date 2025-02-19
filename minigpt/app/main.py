import subprocess
import time


def run_app():
    fastapi_process = subprocess.Popen(["uvicorn", "minigpt.app.fastapi_app:app", "--reload"])

    time.sleep(1)

    streamlit_process = subprocess.Popen(["streamlit", "run", "minigpt/app/streamlit_app.py"])

    fastapi_process.wait()
    streamlit_process.wait()


if __name__ == "__main__":
    run_app()