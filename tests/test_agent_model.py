"""Choosing the agent's model from settings: Bedrock when YOHO_AGENT_MODEL is set, else Ollama."""

from __future__ import annotations

from strands.models.bedrock import BedrockModel
from strands.models.ollama import OllamaModel

from agent.agent_interaction import make_model
from web.config import Settings

REQUIRED = {"FLASK_SECRET_KEY": "s", "GOOGLE_CLIENT_ID": "id", "GOOGLE_CLIENT_SECRET": "secret", "YOHO_DATA_KEYS": "k"}


def test_model_id_selects_bedrock_in_region():
    model = make_model("us.example.model-v1:0", "us-east-1")
    assert isinstance(model, BedrockModel)
    assert model.get_config()["model_id"] == "us.example.model-v1:0"
    assert model.client.meta.region_name == "us-east-1"


def test_no_model_id_falls_back_to_ollama():
    model = make_model(None, "us-east-1")
    assert isinstance(model, OllamaModel)
    assert model.get_config()["model_id"] == "gemma4:e2b"


def test_settings_read_agent_model(monkeypatch):
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("YOHO_AGENT_MODEL", "us.example.model-v1:0")
    assert Settings.from_env().agent_model == "us.example.model-v1:0"
    monkeypatch.setenv("YOHO_AGENT_MODEL", "")
    assert Settings.from_env().agent_model is None
