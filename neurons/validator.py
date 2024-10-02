import os
import sys
import asyncio
import datetime as dt
import wandb
import bittensor as bt
import uvicorn
from pyngrok import ngrok  # Import ngrok from pyngrok

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

from ttm.ttm import MusicGenerationService
from ttm.aimodel import AIModelService


class AIModelController():
    def __init__(self):
        self.aimodel = AIModelService()
        self.music_generation_service = MusicGenerationService()
        self.current_service = self.music_generation_service
        self.last_run_start_time = dt.datetime.now()

    async def run_fastapi_with_ngrok(self, app):
        # Setup ngrok tunnel
        ngrok_tunnel = ngrok.connect(38287, bind_tls=True)
        print('Public URL:', ngrok_tunnel.public_url)
        # Create and start the uvicorn server as a background task
        config = uvicorn.Config(app=app, host="0.0.0.0", port=38287)  # Ensure port matches ngrok's
        server = uvicorn.Server(config)
        # No need to await here, as we want this to run in the background
        task = asyncio.create_task(server.serve())
        return ngrok_tunnel, task  # Returning task if you need to cancel it later


    async def run_services(self):
        while True:
            self.check_and_update_wandb_run()
            await self.music_generation_service.run_async()

    def check_and_update_wandb_run(self):
        # Calculate the time difference between now and the last run start time
        current_time = dt.datetime.now()
        time_diff = current_time - self.last_run_start_time
        # Check if 4 hours have passed since the last run start time
        if time_diff.total_seconds() >= 4 * 3600:  # 4 hours * 3600 seconds/hour
            self.last_run_start_time = current_time  # Update the last run start time to now
            if self.wandb_run:
                wandb.finish()  # End the current run
            self.new_wandb_run()  # Start a new run

    def new_wandb_run(self):
        now = dt.datetime.now()
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = f"Validator-{self.aimodel.uid}-{run_id}"
        commit = self.aimodel.get_git_commit_hash()
        self.wandb_run = wandb.init(
            name=name,
            project="AudioSubnet_Valid",
            entity="subnet16team",
            config={
                "uid": self.aimodel.uid,
                "hotkey": self.aimodel.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "Validator",
                "tao (stake)": self.aimodel.metagraph.neurons[self.aimodel.uid].stake.tao,
                "commit": commit,
            },
            tags=self.aimodel.sys_info,
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

async def setup_and_run(controller):
    tasks = []
    secret_key = os.getenv("AUTH_SECRET_KEY")
    if os.path.exists(os.path.join(project_root, 'app')) and secret_key:
        app = create_app(secret_key)
        # Start FastAPI with ngrok without blocking
        ngrok_tunnel, server_task = await controller.run_fastapi_with_ngrok(app)
        tasks.append(server_task)  # Keep track of the server task if you need to cancel it later
    
    # Start service-related tasks
    service_task = asyncio.create_task(controller.run_services())
    tasks.append(service_task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Cleanup, in case you need to close ngrok or other resources
    ngrok_tunnel.close()

async def main():
    controller = AIModelController()
    await setup_and_run(controller)

if __name__ == "__main__":
    asyncio.run(main())