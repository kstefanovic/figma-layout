module.exports = {
  apps: [
    {
      name: "qwen-model-server",
      cwd: __dirname,
      script: "model_server.py",
      interpreter: ".venv/bin/python",
      autorestart: true,
      max_restarts: 5,
      restart_delay: 5000,
      kill_timeout: 10000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "qwen-backend",
      cwd: __dirname,
      script: "backend.py",
      interpreter: ".venv/bin/python",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      kill_timeout: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
