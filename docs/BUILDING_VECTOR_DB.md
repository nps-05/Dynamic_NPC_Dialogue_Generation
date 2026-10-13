# Building ChromaDB Vector Database

## Overview

The vector database is used for semantic search to find relevant dialogue examples. This guide explains how to build the ChromaDB from CSV files, which is necessary when adding new characters.

## Quick Start

### Build Vector DB for Existing Character

```bash
python build_vector_db.py --csv data/nick_valentine_dialogue.csv
```

The dropdown lists a character only when this command's default database exists next to the CSV. `data/ada_dialogue.csv` becomes `chroma_db_ada/`. Run the command from the project root, and do not pass a different `--db-path` unless you also want to change the discovery code.

This will:
- Read the CSV file
- Generate embeddings using sentence-transformers
- Create a ChromaDB at `chroma_db_nick_valentine`
- Store each line with context and emotion metadata

### Build Vector DB for New Character

```bash
python build_vector_db.py \
  --csv data/new_character_dialogue.csv \
  --db-path chroma_db_new_character \
  --collection new_character
```

## CSV Format Requirements

Your CSV file should have these columns:
- **RESPONSE TEXT** (required) - The dialogue text to embed
- **CONTEXT** (optional, preferred) - One of `investigation`, `combat`, `emotional`, `casual`, `greeting`, `moral_choice`, `location`
- **EMOTION** (optional, preferred) - One of `neutral`, `amused`, `stern`, `concerned`, `irritated`, `somber`, `puzzled`, `angry`, `surprised`
- **SCRIPT NOTES** (optional) - Free-text tone hints used only when CONTEXT/EMOTION are missing
- **CATEGORY**, **TYPE**, **SCENE** (optional) - Extra metadata, also scanned for keywords

If CONTEXT and EMOTION are present, they are stored as-is. Otherwise the builder guesses from notes and the line itself. Guessed rows used to collapse to `casual` / `neutral`, which made filtered search return nothing.

Example CSV structure:
```csv
CATEGORY,TYPE,RESPONSE TEXT,SCRIPT NOTES,SCENE,CONTEXT,EMOTION
Sample,CaseWork,"Start with the last person who saw them.",Stern,missing_person,investigation,stern
Sample,Greeting,"You found the office. That is the easy part.",Neutral,office_hello,greeting,neutral
```

The file shipped in `data/nick_valentine_dialogue.csv` is a short original sample, not a game export. Replace it with your own lines in this shape, then rebuild. Do not commit dialogue extracted from a commercial game.

## Command Line Options

```bash
python build_vector_db.py [OPTIONS]

Options:
  --csv PATH          Path to CSV file with dialogue data (required)
  --db-path PATH      Path where ChromaDB will be stored 
                      (default: chroma_db_<character_name>)
  --collection NAME   Name for ChromaDB collection
                      (default: character name from CSV filename)
  --model MODEL       Sentence transformer model for embeddings
                      (default: all-MiniLM-L6-v2)
```

## Examples

### Example 1: Build for Nick Valentine
```bash
python build_vector_db.py --csv data/nick_valentine_dialogue.csv
```

### Example 2: Build for New Character with Custom Path
```bash
python build_vector_db.py \
  --csv data/piper_dialogue.csv \
  --db-path chroma_db_piper \
  --collection piper
```

### Example 3: Use Different Embedding Model
```bash
python build_vector_db.py \
  --csv data/nick_valentine_dialogue.csv \
  --model all-mpnet-base-v2
```

## How It Works

1. **Loads CSV**: Reads dialogue entries from CSV file
2. **Generates Embeddings**: Uses sentence-transformers to create vector embeddings
3. **Extracts Metadata**: Infers context and emotion from CSV columns
4. **Stores in ChromaDB**: Saves embeddings with metadata for semantic search

## Metadata Extraction

The script automatically extracts metadata:

- **Emotion**: From SCRIPT NOTES column (Happy → amused, Stern → stern, etc.)
- **Context**: From CATEGORY/TYPE (combat, investigation, emotional, greeting, casual)
- **Other**: Category, type, scene from respective columns

## Adding New Characters

To add a new character:

1. **Create CSV file** in `data/` directory:
   ```bash
   data/new_character_dialogue.csv
   ```

2. **Build vector database**:
   ```bash
   python build_vector_db.py \
     --csv data/new_character_dialogue.csv \
     --db-path chroma_db_new_character \
     --collection new_character
   ```

3. **Point the app at the new database.** `app.py` currently opens `chroma_db_nick_valentine`. Change that path (and the collection the agent expects) before relying on another character's index.

## Troubleshooting

### Error: "No collections found in ChromaDB!"
- Make sure you've run `build_vector_db.py` first
- Check that the `--db-path` matches what the agent expects

### Error: "ModuleNotFoundError: No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Error: "ModuleNotFoundError: No module named 'tqdm'"
```bash
pip install tqdm
```

### Slow Embedding Generation
- First run downloads the embedding model (~80MB)
- Large CSV files take time - be patient
- Consider using a smaller embedding model: `--model all-MiniLM-L6-v2`

## Embedding Models

Default: `all-MiniLM-L6-v2` (fast, good quality)

Other options:
- `all-mpnet-base-v2` - Better quality, slower
- `paraphrase-MiniLM-L6-v2` - Good for dialogue similarity
- `all-MiniLM-L12-v2` - Larger, better quality

## File Structure After Building

```
project/
├── chroma_db_nick_valentine/     # Created by build script
│   ├── chroma.sqlite3
│   └── [collection files]
├── chroma_db_new_character/       # For new characters
│   └── ...
└── data/
    ├── nick_valentine_dialogue.csv
    └── new_character_dialogue.csv
```

## Notes

- The script will **overwrite** existing collections with the same name
- Embeddings are generated in batches for efficiency
- Metadata is automatically cleaned (handles 'nan', empty strings, etc.)
- The vector store is persistent - you only need to rebuild if CSV changes



