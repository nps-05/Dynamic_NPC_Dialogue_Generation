import os
import unittest

from agents.base_agent import BaseDialogueAgent
from app import agent_manager, app
from dialogue_processor import DialogueProcessor, infer_retrieval_labels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "nick_valentine_dialogue.csv")


class FakeAgent(BaseDialogueAgent):
    def __init__(self):
        super().__init__("Fake", "fake-model")

    def initialize(self):
        return True

    def generate_response(self, user_input, context="casual", emotion=None, include_examples=True):
        reply = f"Heard you: {user_input}"
        self.add_to_history(user_input, reply, context, emotion)
        return {
            "response": reply,
            "context": context,
            "emotion": emotion,
            "agent": self.name,
            "model": self.model_name,
            "used_examples": include_examples,
        }


class SmokeTests(unittest.TestCase):
    def setUp(self):
        agent_manager.agents.clear()
        agent_manager.active_agent = None
        agent_manager.register_agent("fake", FakeAgent())
        self.client = app.test_client()

    def test_health_and_agent_list(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["agents_count"], 1)
        self.assertEqual(health.get_json()["active_agent"], "fake")

        agents = self.client.get("/api/agents").get_json()["agents"]
        self.assertEqual(agents[0]["model"], "fake-model")
        self.assertTrue(agents[0]["active"])

    def test_generate_requires_input(self):
        response = self.client.post("/api/generate", json={"userInput": "   "})
        self.assertEqual(response.status_code, 400)

    def test_generate_uses_registered_agent(self):
        response = self.client.post(
            "/api/generate",
            json={
                "userInput": "Who saw them last?",
                "context": "investigation",
                "emotion": "stern",
                "includeExamples": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["response"], "Heard you: Who saw them last?")
        self.assertEqual(body["model"], "fake-model")
        self.assertEqual(body["context"], "investigation")
        self.assertFalse(body["used_examples"])

    def test_history_reset_and_export(self):
        self.client.post("/api/generate", json={"userInput": "Hello"})
        history = self.client.get("/api/history").get_json()["history"]
        self.assertEqual(len(history), 1)

        exported = self.client.get("/api/export").get_json()
        self.assertEqual(exported["model"], "fake-model")
        self.assertEqual(len(exported["conversation"]), 1)

        reset = self.client.post("/api/reset")
        self.assertTrue(reset.get_json()["success"])
        self.assertEqual(self.client.get("/api/history").get_json()["history"], [])

    def test_sample_csv_is_original_and_labeled(self):
        with open(CSV_PATH, encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn("Hell of a game", text)
        self.assertNotIn("Myrna", text)
        self.assertLess(len(text.splitlines()), 40)

        processor = DialogueProcessor(CSV_PATH)
        self.assertGreaterEqual(len(processor.dialogues), 8)
        labeled = infer_retrieval_labels(processor.dialogues[0])
        self.assertEqual(labeled["context"], "investigation")
        self.assertEqual(labeled["emotion"], "stern")

    def test_explicit_labels_override_keywords(self):
        labels = infer_retrieval_labels(
            {
                "RESPONSE TEXT": "Hello there.",
                "SCRIPT NOTES": "Happy",
                "CONTEXT": "investigation",
                "EMOTION": "stern",
            }
        )
        self.assertEqual(labels["context"], "investigation")
        self.assertEqual(labels["emotion"], "stern")

    def test_keywords_fill_missing_labels(self):
        labels = infer_retrieval_labels(
            {
                "RESPONSE TEXT": "Check the witness statement again.",
                "SCRIPT NOTES": "Puzzled",
                "CATEGORY": "Scene",
            }
        )
        self.assertEqual(labels["context"], "investigation")
        self.assertEqual(labels["emotion"], "puzzled")


if __name__ == "__main__":
    unittest.main()
