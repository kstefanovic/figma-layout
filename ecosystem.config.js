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
        BACKEND_PORT: "20001",
        TOP_LEVEL_LAYOUT_ENGINE: "ralf",
        TOP_LEVEL_LAYOUT_RALF_CHECKPOINT:
          "layout_training/checkpoints/top_level_layout_ralf_v1.pt",
        TOP_LEVEL_LAYOUT_RALF_INDEX:
          "layout_training/data/layout_records/top_level_retrieval_index.pt",
        TOP_LEVEL_LAYOUT_RALF_RECORDS:
          "layout_training/data/layout_records/top_level_records.jsonl",
      },
    },
  ],
};
