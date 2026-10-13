import os
import tempfile
import unittest

import app as app_module
from agents.base_agent import BaseDialogueAgent
from characters import discover_characters, personality_for, public_character


class CharacterDiscoveryTests(unittest.TestCase):
    def test_csv_without_database_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data")
            os.makedirs(data)
            with open(os.path.join(data, "ada_dialogue.csv"), "w", encoding="utf-8") as handle:
                handle.write("RESPONSE TEXT\nHello.\n")
            self.assertEqual(discover_characters(tmp), [])

    def test_pair_is_listed_and_prompt_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data")
            db = os.path.join(tmp, "chroma_db_ada")
            os.makedirs(data)
            os.makedirs(db)
            with open(os.path.join(data, "ada_dialogue.csv"), "w", encoding="utf-8") as handle:
                handle.write("RESPONSE TEXT\nHello.\n")
            with open(os.path.join(db, "chroma.sqlite3"), "w", encoding="utf-8") as handle:
                handle.write("")

            found = discover_characters(tmp)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["id"], "ada")
            self.assertEqual(found[0]["name"], "Ada")
            self.assertIsNone(found[0]["prompt_path"])
            self.assertFalse(public_character(found[0])["hasPrompt"])
            self.assertIn("You are Ada", personality_for(found[0]))

            prompt_dir = os.path.join(tmp, "characters", "ada")
            os.makedirs(prompt_dir)
            with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as handle:
                handle.write("Speak like a dockworker.")
            found = discover_characters(tmp)
            self.assertTrue(public_character(found[0])["hasPrompt"])
            self.assertEqual(personality_for(found[0]), "Speak like a dockworker.")


class CharacterSwitchTests(unittest.TestCase):
    def test_switch_rejects_a_character_without_a_database(self):
        class Switchable(BaseDialogueAgent):
            def __init__(self):
                super().__init__("Fake", "fake-model")
                self.character_id = None

            def initialize(self):
                return True

            def generate_response(self, user_input, context="casual", emotion=None, include_examples=True):
                return {"response": "ok"}

            def use_character(self, character):
                self.character_id = character["id"]
                self.reset_conversation()

        app_module.agent_manager.agents.clear()
        app_module.agent_manager.active_agent = None
        app_module.agent_manager.register_agent("fake", Switchable())
        client = app_module.app.test_client()

        missing = client.post("/api/characters/switch", json={"characterId": "not-built"})
        self.assertEqual(missing.status_code, 404)

        listed = client.get("/api/characters")
        self.assertEqual(listed.status_code, 200)
        self.assertIsInstance(listed.get_json()["characters"], list)


if __name__ == "__main__":
    unittest.main()
