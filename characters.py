"""Find characters that have both a dialogue CSV and a built vector database."""

import os
from typing import Dict, List, Optional


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def character_id_from_filename(filename: str) -> str:
    """Match build_vector_db.py: nick_valentine_dialogue.csv -> nick_valentine."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.endswith("_dialogue"):
        stem = stem[: -len("_dialogue")]
    return stem


def display_name(character_id: str) -> str:
    return character_id.replace("_", " ").strip().title() or character_id


def db_dir_for(character_id: str, root: Optional[str] = None) -> str:
    root = root or project_root()
    return os.path.join(root, f"chroma_db_{character_id}")


def discover_characters(root: Optional[str] = None) -> List[Dict]:
    """Return characters whose CSV and Chroma database both exist.

    A CSV alone is not enough. The database is the local index produced by
    build_vector_db.py, and it is not shipped with the repo.
    """
    root = root or project_root()
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        return []

    found = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".csv"):
            continue
        character_id = character_id_from_filename(filename)
        if not character_id:
            continue
        csv_path = os.path.join(data_dir, filename)
        db_path = db_dir_for(character_id, root)
        if not os.path.isfile(os.path.join(db_path, "chroma.sqlite3")):
            continue
        prompt_path = os.path.join(root, "characters", character_id, "prompt.txt")
        found.append(
            {
                "id": character_id,
                "name": display_name(character_id),
                "csv_path": csv_path,
                "db_path": db_path,
                "prompt_path": prompt_path if os.path.isfile(prompt_path) else None,
            }
        )
    return found


def public_character(character: Dict) -> Dict:
    return {
        "id": character["id"],
        "name": character["name"],
        "hasPrompt": bool(character.get("prompt_path")),
    }


def personality_for(character: Dict) -> str:
    prompt_path = character.get("prompt_path")
    if prompt_path and os.path.isfile(prompt_path):
        with open(prompt_path, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text:
            return text

    name = character["name"]
    return (
        f"You are {name}. Stay in character.\n\n"
        "SPEAKING STYLE:\n"
        "- Use short, direct sentences\n"
        "- Never mention that you are an AI or a language model\n"
        "- Keep responses to 1-2 sentences\n"
        "- Do not explain the character. Speak as them."
    )
